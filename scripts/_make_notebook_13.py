from pathlib import Path
import nbformat
from nbformat import v4 as nbf

cells=[]
def md(text): cells.append(nbf.new_markdown_cell(text))
def code(text): cells.append(nbf.new_code_cell(text))

md("""# Warm-start RQAOA: ансамбль стартов

Ноутбук показывает:
- карточку `warm_start_ensemble_rqaoa_cpu_v1`;
- global-flip dedup и multiplicities;
- два реальных warm-state child на текущем графе;
- равную смесь физических корреляторов;
- отрицательное ребро, offset и lift;
- разницу mixture и coherent sum;
- квоты общего бюджета;
- реальный GW batch;
- малую серию из 12 trials.

Первый сценарий не требует CVXPY. GW-секция запускается в classical-профиле.
""")

code("""from pathlib import Path
import copy, json, math, os, tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path.cwd()
if not (ROOT/"configs").exists(): ROOT=ROOT.parent
os.chdir(ROOT)

from QA_max_cut_lib import WeightedGraph, tools
from QA_max_cut_lib.core.budget import ShotBudget
from QA_max_cut_lib.laboratory import WarmStartRQAOACPU, get_variant
from QA_max_cut_lib.solvers.warm_start_ensemble import WarmStartEnsembleCorrelationSource

print(get_variant("warm_start_ensemble_rqaoa_cpu_v1")["status"])
assert get_variant("warm_start_ensemble_rqaoa_cpu_v1")["runnable"]
""")

md("""## 1. Контроль A: два старта

A=(+,+,−,+), B=(+,−,+,+) в порядке `(10,30,80,101)`. A и −A после
global-flip dedup дают один ключ; multiplicity A равна 3, B — 1.
""")

code("""graph=WeightedGraph([(10,30,2),(10,80,3),(30,80,5)],nodes=[10,30,80,101])
A={10:1,30:1,80:-1,101:1}; B={10:1,30:-1,80:1,101:1}
record={"graph":graph.snapshot(),"assignments":[
 {"candidate_id":"A","spins":A},
 {"candidate_id":"-A","spins":{k:-v for k,v in A.items()}},
 {"candidate_id":"B","spins":B},
 {"candidate_id":"A2","spins":A},
]}
cfg={
 "schema_version":1,"type":"laboratory.warm_start_rqaoa_cpu","seed":20261102,
 "parameters":{"finish_at":3,"max_reductions":1},
 "correlation_source":{
  "type":"warm_start_ensemble_cpu",
  "parameters":{"layers":1,"max_vertices":18,"dtype":"complex128"},
  "candidate_source":{"type":"provided_by_graph","parameters":{"records":[record]}},
  "starts":{"max_starts":2,"deduplication":"global_flip","on_shortage":"use_available","ranking":"current_cut_then_canonical_spins"},
  "warm_start":{"epsilon":0.25,"mixer":{"type":"rounded_modified"}},
  "objective":{"type":"mean_cut"},
  "estimation":{"mode":"exact","shots_per_evaluation":32},
  "optimizer":{"type":"adam","parameters":{"max_steps":0,"learning_rate":0.05,"gradient_tolerance":0.0},"gradient":{"type":"central_difference","parameters":{"step":0.01}}},
  "initialization":{"type":"fixed_angles","parameters":{"values":[0.0,math.pi/2]}},
  "step_initialization":"restart",
  "correlations":{"mode":"exact","aggregation":"uniform_components"},
 },
 "selection":{"type":"max_abs_correlation","parameters":{"pairs":"edges","zero_sign":1}},
 "finisher":{"type":"exact_cpu","parameters":{"max_vertices":16}},
 "limits":{"shots_total":0},
 "collect":{"reduction_history":True,"parameter_trajectory":False,"local_optimizer_history":False,"correlation_tables":True,"final_reduction_state":True},
 "output":{"report":0,"logs":0},
}
solver=WarmStartRQAOACPU(cfg)
solution=solver.solve(graph)
history=solver.warm_start_history[0]
print("solution",solution)
print("C",solver.cut_value,"b",solver.cut_offset,"E_Ising",solver.ising_constant)
print("unique",history["source"]["unique"],"multiplicities",[c["multiplicity"] for c in history["children"]])
assert solver.cut_value==8.0 and solver.cut_offset==8.0 and solver.ising_constant==-7.5
assert [c["multiplicity"] for c in history["children"]]==[3,1]
""")

md("""## 2. Смесь vs coherent sum

Для двух кубитов равная смесь `ψ1,ψ2` даёт равномерное P и c=0.
Нормированная сумма амплитуд даёт другой ответ; это ловит ошибочную
coherent aggregation.
""")

code("""psi1=np.array([1,1,1,1],dtype=complex)/2
psi2=np.array([1,-1,-1,1],dtype=complex)/2
p1=np.abs(psi1)**2; p2=np.abs(psi2)**2
p_mix=(p1+p2)/2
coherent=psi1+psi2; coherent=coherent/np.linalg.norm(coherent)
p_coh=np.abs(coherent)**2
print("P mixture",p_mix,"P coherent",p_coh)
assert np.allclose(p_mix,0.25)
assert not np.allclose(p_coh,p_mix)
""")

md("""## 3. Квоты общего бюджета

p=1, M_train=32, M_corr=16, max_steps=2, B=368:
для K=3 получается [1,1,0] и reserve 48; для K=2 — [1,1].
""")

code("""qcfg=copy.deepcopy(cfg["correlation_source"])
qcfg["estimation"]={"mode":"shots","shots_per_evaluation":32}
qcfg["optimizer"]["parameters"]["max_steps"]=2
qcfg["correlations"]={"mode":"shots","shots":16,"aggregation":"uniform_components"}
qsrc=WarmStartEnsembleCorrelationSource(qcfg,seed=1)
q3,r3=qsrc._quotas(ShotBudget(368),3,qsrc.config)
q2,r2=qsrc._quotas(ShotBudget(368),2,qsrc.config)
print(q3,r3,q2,r2)
assert q3==[1,1,0] and r3==48 and q2==[1,1] and r2==32
""")

md("""## 4. Реальный GW batch

Один SDP, L округлений, без K отдельных SDP. Эта секция требует CVXPY/SCS.
""")

code("""try:
    gwcfg=copy.deepcopy(cfg)
    gwcfg["correlation_source"]["candidate_source"]={"type":"gw_rounds_cpu","parameters":{"num_hyperplanes":8,"solver":"SCS","allow_inaccurate":False}}
    gwcfg["output"]={"report":0,"logs":0}
    gw=WarmStartRQAOACPU(gwcfg); gw.solve(graph)
    source=gw.warm_start_history[0]["source"]
    print("SDP calls",source["source_cost"]["sdp_calls"],"rounds",source["rounds_completed"])
    assert source["source_cost"]["sdp_calls"]==1
except ImportError as exc:
    print("GW section skipped:",exc)
""")

md("""## 5. Серия 12 trials

Два графа × два algorithm seed × три recipe: обычный RQAOA, WS с K=1,
WS с K=2. Это 12 trials, не 12 quantum children.
""")

code("""spec=tools.load_json("experiments/benchmark_warm_start_rqaoa.json")
spec.pop("save",None)
series=tools.ExperimentRunner(spec).run()
print(series.summary)
print(json.dumps(series.analysis["by_solver"],indent=2)[:2000])
assert series.analysis["planned"]==12 and series.analysis["valid"]==12
""")

md("""## 6. Сравнение и график

График использует только сохранённые записи серии; solver/source повторно не
запускаются. Различия качества — наблюдение, не обещание преимущества.
""")

code("""names=sorted(series.analysis["by_solver"])
means=[series.analysis["by_solver"][n]["cut"]["mean"] for n in names]
fig,ax=plt.subplots(figsize=(9,4))
ax.bar(names,means,color=["#64748b","#2563eb","#7c3aed"])
ax.set_ylabel("mean C_original")
ax.set_title("Small WS-RQAOA series: solver-only quality")
ax.grid(axis="y",alpha=.25)
fig.tight_layout()
out=Path("docs/notebook_previews/13_warm_start_rqaoa.png")
out.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(out,dpi=140,facecolor="white")
plt.close(fig)
print("saved",out)
""")

md("""## Итог

- A: U=2, multiplicities 3/1, C=8, b=8, E_Ising=-7.5.
- Смесь корреляций — статистическая, не coherent sum.
- Квоты K-зависимы и не зависят от worker count.
- Реальный GW: один SDP и L ограниченных округлений.
- Серия: 12 trials, 0 failed (в этой среде).
- Checkpoint/resume для multi-start RQAOA в a12 не заявлен.
""")

nb=nbf.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python 3.12 (QA_max_cut_lib)","language":"python","name":"qa-max-cut-lib"},"language_info":{"name":"python","version":"3.12"}})
Path("notebooks/13_warm_start_rqaoa.ipynb").write_text(nbformat.writes(nb),encoding="utf-8")
print("wrote notebook 13")
