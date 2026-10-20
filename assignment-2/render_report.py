"""Render the audited final campaign without changing training or attack artifacts.

Historical sources/results remain under /work; this updates the existing report
structure and citations for the regularized, validation-controlled protocol.
"""
from pathlib import Path
import argparse,json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
KEYS=['config_id','clients','batch_size','local_epochs','learning_rate']
PLOT_SEED=684


def baseline(frame, history=False):
    lr='initial_learning_rate' if history else 'learning_rate'
    return frame[(frame.batch_size==32)&(frame.local_epochs==1)&(frame[lr]==.05)]


def mean_sd(frame,key):
    return f"{frame[key].mean():.4f} $\\pm$ {frame[key].std(ddof=1):.4f}"


def diagnosis(history):
    rows=[]
    for (cid,seed),g in history.groupby(['config_id','seed']):
        g=g.sort_values('round');loss=g.train_cross_entropy.to_numpy()
        window=np.convolve(loss,np.ones(10)/10,mode='valid')[-20:]
        d=max(float(np.median(window)),1e-12)
        change=float(np.polyfit(np.arange(20),window,1)[0]*19)
        rates=g.learning_rate.to_numpy();fixed=int(np.argmax(rates[::-1]!=rates[-1])) if (rates!=rates[-1]).any() else len(g)
        rows.append(dict(config_id=cid,seed=seed,rounds=len(g),smoothed_window_range=float(np.ptp(window)),
                         signed_fitted_window_change=change,absolute_fitted_window_change=abs(change),
                         strict_range_limit=.005*d,strict_trend_limit=.001*d,
                         practical_range_limit=max(.0005,.005*d),practical_trend_limit=max(.0001,.001*d),
                         final_learning_rate=rates[-1],fixed_lr_rounds=fixed,
                         endpoint_training_loss_stable=bool(g.training_loss_stable.iloc[-1]),
                         endpoint_strict_training_loss_stable=bool(g.strict_training_loss_stable.iloc[-1])))
    return pd.DataFrame(rows)


def figures(destination,status,history,attack_per_seed):
    """Show seed 684 only; local attack points still average clients within that seed."""
    output=destination/'figures';output.mkdir(exist_ok=True)
    status=status[status.seed==PLOT_SEED].copy()
    history=history[history.seed==PLOT_SEED].copy()
    attacks=attack_per_seed[attack_per_seed.seed==PLOT_SEED].copy()
    if len(status)!=28 or len(attacks)!=56:
        raise ValueError('Incomplete seed-684 figure inputs')
    colors={1:'#2563eb',10:'#d97706',20:'#059669',50:'#9333ea'}
    fig,axes=plt.subplots(1,3,figsize=(10,2.25),layout='constrained')
    for k,v in baseline(history,True).groupby('clients'):
        v=v.sort_values('round')
        sr=int(status.loc[status.config_id==v.config_id.iloc[0],'selected_round'].iloc[0])
        selected=v.loc[v['round']==sr].iloc[0]
        v=v[v['round']>=10]
        for ax,col in zip(axes,['train_cross_entropy','validation_cross_entropy','test_auroc']):
            ax.plot(v['round'],v[col],color=colors[k],lw=1.4,label=f'K={k}')
            ax.scatter(sr,selected[col],color=colors[k],s=16)
        axes[1].plot(v['round'],v.test_cross_entropy,color=colors[k],lw=1.2,ls='--')
    for ax,title in zip(axes,['Training CE','Validation CE (solid), test CE (dashed)','Test AUROC']):
        ax.set_title(title,fontsize=9);ax.set_xlabel('Round',fontsize=9);ax.tick_params(labelsize=8);ax.grid(alpha=.2)
    axes[0].set_ylim(.27,.37);axes[1].set_ylim(.30,.37)
    axes[0].legend(fontsize=7,ncol=2)
    fig.suptitle('Baseline trajectories — seed 684',fontsize=10)
    fig.savefig(output/'training.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(4.6,2.7),layout='constrained')
    for kind,color in [('global','#2563eb'),('local','#d97706')]:
        v=baseline(attacks);v=v[v.model_type==kind].sort_values('clients')
        if len(v)!=4:raise ValueError('Missing baseline attack points')
        ax.plot(np.arange(4),v.auc,marker='o',color=color,label=kind.capitalize())
    ax.axhline(.5,color='gray',lw=1,ls='--');ax.set_xticks(range(4),['1','10','20','50'])
    ax.set_xlabel('Clients');ax.set_ylabel('First-round LOSS AUROC');ax.set_title('Seed 684; local = client mean',fontsize=10)
    ax.set_ylim(.48,.535);ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(output/'attack_auc.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(9,2.7),layout='constrained')
    for ax,(parameter,values) in zip(axes,[('batch_size',[16,32,64]),('local_epochs',[1,3,5]),('learning_rate',[.01,.05,.1])]):
        for k in [1,10,20,50]:
            v=status[status.clients==k]
            for key,value in {'batch_size':32,'local_epochs':1,'learning_rate':.05}.items():
                if key!=parameter:v=v[v[key]==value]
            if len(v)!=3 or v[parameter].duplicated().any():raise ValueError('Missing parameter-effect points')
            points=v.set_index(parameter).test_auroc.reindex(values)
            ax.plot(range(3),points,marker='o',color=colors[k],label=f'K={k}')
        ax.set_xticks(range(3),[str(x) for x in values])
        label='initial learning rate' if parameter=='learning_rate' else parameter.replace('_',' ')
        ax.set_xlabel(label);ax.set_ylabel('Selected test AUROC');ax.grid(alpha=.2)
        ax.set_ylim(.54,.76)
    axes[0].legend(fontsize=7,ncol=2);fig.suptitle('Parameter effects — seed 684',fontsize=10)
    fig.savefig(output/'parameter_effects.png',dpi=180);plt.close(fig)


def render_final(destination):
    r=destination/'results';coverage=json.loads((r/'coverage.json').read_text())
    if not coverage.get('complete') or not coverage.get('passed') or coverage.get('errors'):
        raise ValueError('Refuse report generation without complete independent audit')
    if (coverage['finalized_runs'],coverage['verified_model_evaluations'])!=(84,1785):raise ValueError('Wrong grid coverage')
    status=pd.read_csv(r/'run_status.csv');history=pd.read_csv(r/'round_metrics.csv');attacks=pd.read_csv(r/'attack_summary.csv')
    if len(status)!=84 or len(history)!=16088 or status[['config_id','seed']].duplicated().any():raise ValueError('Missing/duplicate final artifacts')
    if status.state.value_counts().to_dict()!=coverage['counts']:raise ValueError('Stopping count mismatch')
    b=baseline(status);means=status.groupby(KEYS).mean(numeric_only=True).reset_index()
    def setting(k,parameter=None,value=None):
        target=dict(clients=k,batch_size=32,local_epochs=1,learning_rate=.05)
        if parameter:target[parameter]=value
        q=means
        for key,value in target.items():q=q[q[key]==value]
        if len(q)!=1:raise ValueError('Missing setting')
        return q.iloc[0]
    performance=[];effects=[];attack_rows=[]
    for k in [1,10,20,50]:
        v=b[b.clients==k]
        performance.append(f"{k} & {v['rounds'].min()}--{v['rounds'].max()} & "+' & '.join(mean_sd(v,key) for key in ['train_cross_entropy','test_cross_entropy','test_auroc','test_calibrated_balanced_accuracy','test_calibrated_sensitivity'])+r" \\")
        changes=[]
        for p,lo,hi in [('batch_size',16,64),('local_epochs',1,5),('learning_rate',.01,.1)]:
            low,high=setting(k,p,lo),setting(k,p,hi)
            changes.append(f"{high.train_cross_entropy-low.train_cross_entropy:+.4f} / {high.test_auroc-low.test_auroc:+.4f}")
        effects.append(str(k)+' & '+' & '.join(changes)+r" \\")
        for kind in ['global','local']:
            v=baseline(attacks);v=v[(v.clients==k)&(v.model_type==kind)].iloc[0]
            cols=['oracle_threshold_balanced_accuracy','auc','tpr_at_1pct_fpr','tpr_at_5pct_fpr','tpr_at_10pct_fpr']
            attack_rows.append(f"{k} & {kind.title()} & "+' & '.join(f"{v[c+'_mean']:.4f} $\\pm$ {v[c+'_std']:.4f}" for c in cols)+r" \\")
    d=diagnosis(history);d.to_csv(r/'training_diagnostics.csv',index=False)
    cap=status[status.state=='capped'];cd=d.merge(cap[['config_id','seed']],on=['config_id','seed'])
    stable=status[status.state=='loss_stable']
    stopping=(r"\textbf{72 of 84 runs reached loss stability; nine stopped on validation patience and three reached the cap.} "
              f"The stable runs stopped at rounds {stable['rounds'].min()}--{stable['rounds'].max()}, and all also passed the original relative-only loss bounds. "
              "Every K=10/20/50 run was stable. All nine early stops and three caps were K=1 variants. "
              "The capped settings were B=16/LR=0.05 (seeds 684/685) and B=32/LR=0.1 (seed 684), with E=1. "
              f"Their last-window fitted CE decrease was {abs(cd.signed_fitted_window_change).min():.6f}--{abs(cd.signed_fitted_window_change).max():.6f}, "
              f"above their trend limits of {cd.strict_trend_limit.min():.6f}--{cd.strict_trend_limit.max():.6f}. "
              r"\textbf{These three runs were still improving slowly, so we retain their unconverged labels.} "
              "Early stopping does not establish optimization convergence.")
    k1,k50=setting(1),setting(50);bl,bh=setting(50,'batch_size',16),setting(50,'batch_size',64)
    e1,e3,e5=setting(50,'local_epochs',1),setting(50,'local_epochs',3),setting(50,'local_epochs',5)
    ll,lh=setting(50,'learning_rate',.01),setting(50,'learning_rate',.1)
    efficiency=(r"\textbf{More clients meant shorter local update paths and slower learning in the baseline.} "
                f"Increasing K from 1 to 50 raised mean stop rounds from {k1['rounds']:.0f} to {k50['rounds']:.0f} and endpoint training CE "
                f"from {k1.endpoint_train_cross_entropy:.4f} to {k50.endpoint_train_cross_entropy:.4f}; selected test AUROC fell from {k1.test_auroc:.4f} to {k50.test_auroc:.4f}. "
                "Each client takes 113 versus three steps per baseline round. Short local paths change the optimization trajectory.\n\n"
                r"\textbf{Larger batches reduced computation but weakened fitting and discrimination.} "
                f"At K=50, increasing B from 16 to 64 reduced mean total steps from {bl.cumulative_optimizer_steps/1000:.1f}k to {bh.cumulative_optimizer_steps/1000:.1f}k "
                f"and recorded time from {bl.elapsed_seconds:.0f} to {bh.elapsed_seconds:.0f} s. Selected training CE rose from {bl.train_cross_entropy:.4f} to {bh.train_cross_entropy:.4f}, "
                f"while test AUROC fell from {bl.test_auroc:.4f} to {bh.test_auroc:.4f}.\n\n"
                r"\textbf{More local epochs saved communication rounds, at a substantial local-computation cost.} "
                f"At K=50, E=1/3/5 used {e1['rounds']:.1f}/{e3['rounds']:.1f}/{e5['rounds']:.1f} mean rounds, "
                f"{e1.cumulative_optimizer_steps/1000:.1f}/{e3.cumulative_optimizer_steps/1000:.1f}/{e5.cumulative_optimizer_steps/1000:.1f}k steps, "
                f"and {e1.cumulative_processed_examples/1e6:.2f}/{e3.cumulative_processed_examples/1e6:.2f}/{e5.cumulative_processed_examples/1e6:.2f} million processed examples. "
                f"Time increased from {e1.elapsed_seconds:.0f} to {e5.elapsed_seconds:.0f} s, while AUROC improved from {e1.test_auroc:.4f} to {e5.test_auroc:.4f}.\n\n"
                r"\textbf{A small initial learning rate could produce stable but underfit models.} "
                f"At K=50, LR 0.01 versus 0.1 gave mean selected training CE {ll.train_cross_entropy:.4f} versus {lh.train_cross_entropy:.4f} "
                f"and test AUROC {ll.test_auroc:.4f} versus {lh.test_auroc:.4f}, despite similar mean stop rounds ({ll['rounds']:.0f}/{lh['rounds']:.0f}). "
                "More local work or LR did not guarantee K=1 convergence.")
    # Historical means were verified before replacing the primary tables.
    prior=json.loads((r/'final_report_provenance.json').read_text())['original_k1_baseline']
    generalization=(r"\textbf{Regularization and validation selection substantially reduced severe overfitting.} "
                    f"The original fixed-300-round K=1 baseline had mean train/test CE {prior['train_cross_entropy']:.4f}/{prior['test_cross_entropy']:.4f}; "
                    f"the selected model now has {k1.train_cross_entropy:.4f}/{k1.test_cross_entropy:.4f}. "
                    f"Across the full grid, selected test CE is {status.test_cross_entropy.min():.4f}--{status.test_cross_entropy.max():.4f}; "
                    f"the largest validation-minus-training and test-minus-training gaps are {(status.validation_cross_entropy-status.train_cross_entropy).max():.4f} "
                    f"and {(status.test_cross_entropy-status.train_cross_entropy).max():.4f}. Some K=1 overfitting remains.\n\n"
                    r"\textbf{High accuracy alone would hide a clinically important failure.} "
                    f"At the default threshold, {(status.test_balanced_accuracy==.5).sum()}/84 selected models predict only negatives. "
                    r"Their 88.78\% accuracy simply matches the negative-class prevalence: BA is 0.5 and readmission sensitivity is zero. "
                    f"Validation-selected thresholds produce both classes in all 84 models. Baseline mean BA is {b.groupby('clients').test_calibrated_balanced_accuracy.mean().min():.4f}--{b.groupby('clients').test_calibrated_balanced_accuracy.mean().max():.4f}, "
                    f"with recall {100*b.groupby('clients').test_calibrated_sensitivity.mean().min():.1f}--{100*b.groupby('clients').test_calibrated_sensitivity.mean().max():.1f}"
                    r"\%. This improves sensitivity at the cost of specificity; it does not improve AUROC or establish clinical usefulness.")
    av=baseline(attacks);am=av.set_index(['clients','model_type'])
    glob=attacks[attacks.model_type=='global'];local=attacks[(attacks.clients==50)&(attacks.model_type=='local')].set_index(['batch_size','local_epochs','learning_rate'])
    lo,hi=local.loc[(32,1,.05)],local.loc[(32,5,.05)]
    leakage=(r"\textbf{Local LOSS attacks were slightly stronger, but both local and global attacks remained weak.} "
             f"Baseline global AUROC is {av[av.model_type=='global'].auc_mean.min():.4f}--{av[av.model_type=='global'].auc_mean.max():.4f}; "
             f"local AUROC is {av[av.model_type=='local'].auc_mean.min():.4f}--{av[av.model_type=='local'].auc_mean.max():.4f}. "
             f"At K=50, the local-minus-global difference is {am.loc[(50,'local'),'auc_mean']-am.loc[(50,'global'),'auc_mean']:.4f}.\n\n"
             r"\textbf{More local work increased the small membership signal.} "
             f"At K=50, E=1 to 5 raised local AUROC from {lo.auc_mean:.4f} to {hi.auc_mean:.4f} and TPR at 1"
             r"\% FPR from " +f"{100*lo.tpr_at_1pct_fpr_mean:.2f}"+r"\% to "+f"{100*hi.tpr_at_1pct_fpr_mean:.2f}"+r"\%. "
             f"B=16 to 64 lowered local AUROC from {local.loc[(16,1,.05),'auc_mean']:.4f} to {local.loc[(64,1,.05),'auc_mean']:.4f}; "
             f"initial LR 0.01 to 0.1 raised it from {local.loc[(32,1,.01),'auc_mean']:.4f} to {local.loc[(32,1,.1),'auc_mean']:.4f}. "
             f"Global configuration means stayed within {glob.auc_mean.min():.4f}--{glob.auc_mean.max():.4f}.")
    replacements=dict(COUNTS=f"{coverage['counts']['loss_stable']} loss-stable endpoints, {coverage['counts']['early_stopped']} validation early stops and {coverage['counts']['capped']} capped runs",
                      PERFORMANCE='\n'.join(performance),EFFECTS='\n'.join(effects),ATTACKS='\n'.join(attack_rows),
                      STOPPING=stopping,EFFICIENCY=efficiency,GENERALIZATION=generalization,LEAKAGE=leakage)
    tex=REPORT_TEMPLATE
    for key,value in replacements.items():tex=tex.replace('@@'+key+'@@',value)
    if '@@' in tex:raise ValueError('Unrendered report fields')
    (destination/'assignment_2_report.tex').write_text(tex)
    (destination/'assignment_2_slide.tex').write_text(SLIDE_TEMPLATE)
    numeric=status.groupby(KEYS).agg({c:['mean','std'] for c in ['rounds','selected_round','train_cross_entropy','test_cross_entropy','test_auroc','test_calibrated_balanced_accuracy','cumulative_optimizer_steps','cumulative_processed_examples','elapsed_seconds']})
    numeric.columns=['_'.join(c) for c in numeric.columns];numeric.reset_index().to_csv(r/'parameter_summary.csv',index=False)
    figures(destination,status,history,pd.read_csv(r/'attack_metrics_per_seed.csv'))
    print('Rendered final report/slide and three figures from 84 verified runs; stopping states preserved.')

REPORT_TEMPLATE = r"""\documentclass[10pt]{article}
\usepackage[margin=0.7in]{geometry}
\usepackage{amsmath,booktabs,graphicx,microtype,hyperref,float}
\hypersetup{colorlinks=true,urlcolor=blue,citecolor=blue}
\graphicspath{{figures/}}
\setlength{\parskip}{3pt}
\widowpenalty=10000
\clubpenalty=10000
\title{Assignment 2: Federated Learning and Membership Inference}
\author{Team 08 \\ Assignment lead: Chetan Kodand Reddy Madadi \\ Team members: Bharat Govil, Naveen Jarpla, Vishnu Fnu}
\date{Fall 2026}
\begin{document}
\begin{center}
{\Large\bfseries Assignment 2: Federated Learning and Membership Inference\par}
\vspace{4pt}
\textbf{Team 08} \hfill Fall 2026\\
Assignment lead: Chetan Kodand Reddy Madadi\\
Team members: Bharat Govil, Naveen Jarpla, Vishnu Fnu
\end{center}
\begin{abstract}
\textbf{The assignment asks us to implement FedAvg, study how its settings affect
training, test first-round membership inference, and assess whether FL is
practical for our healthcare startup.} We use 9,000 synthetic readmission
encounters, reusing Assignment 1's synthetic data generation pipeline. We compare
28 configurations over three seeds and apply LOSS to every client and global model. \textbf{@@COUNTS@@.}
The attacks show weak membership discrimination. We distinguish training-loss
stability from selection of a model that generalizes, and discuss what these
results mean for a healthcare startup.
\end{abstract}

\section{Task A: Data, FedAvg, and Experimental Design}
\textbf{Task A asks us to implement FedAvg, vary its settings, train until convergence,
and report test performance after each round.}
Our use case is predicting 30-day readmission at discharge across hospitals.
\textbf{We reused the synthetic data generation pipeline from Assignment 1},
including its cleaning and mixed TabDDPM approach \cite{tabddpm}, based on the
public UCI Diabetes 130-US Hospitals dataset (CC BY 4.0) \cite{uci}.
We use a frozen dataset of 9,000 encounters; 11.22\% have the positive label.
The CSV, schema, integrity audits and hashes accompany the code.
\textbf{Synthetic labels are not verified clinical outcomes, and synthetic data
alone provide no formal privacy guarantee.}

A shared preprocessor, fitted only on public generator seed records, applies
median imputation, numerical standardization and fixed-schema one-hot encoding.
All clients therefore use the same feature coordinates. Synthetic test records
fit no transformation; identifiers and the raw outcome field are excluded.

For seeds 684/685/686, a stratified 50/50 split gives 4,500 training and 4,500
test records. Reserving 900 stratified validation records within the training
half leaves 3,600 for SGD. A fixed shuffle divides them equally and IID among
$K\in\{1,10,20,50\}$ clients (3,600/360/180/72 records each). Splits stay fixed
across settings, with natural finite-sample class variation.

\textbf{Centrally accessible synthetic validation data are an assumption of this
simulation.} The simulated server uses the same 900-record validation set for
LR scheduling, checkpoint selection and prediction-threshold calibration.
A real deployment would need an appropriate validation mechanism, such as an
authorized shared validation set or federated evaluation of locally held-out
data. We do not implement that deployment mechanism here.

\textbf{FedAvg implementation.} The MLP has ReLU hidden widths 128/64 and two
logits. All clients participate every round, using unweighted CE and SGD without
momentum, with L2 weight decay 0.01 on all parameters. Partial batches are kept
at the same LR.
\textbf{Clients start from identical global weights; the server averages their updates:}
\[
W_{r+1}=\sum_{i=1}^{K}\frac{n_i}{\sum_j n_j}W_{i,r+1},
\]
which is the arithmetic mean for equal partitions \cite{fedavg}.
$K=1$ matches centralized SGD with the same epoch grouping and shuffle stream.

\textbf{Parameter study.} Baseline batch size is 32, local epochs one, initial LR 0.05. One parameter at a
time varies over batches $\{16,32,64\}$, epochs $\{1,3,5\}$, and initial LRs
$\{0.01,0.05,0.1\}$. Deduplication gives 28 configurations and 84 seeded runs;
interactions are not tested. \textbf{All plots use seed 684; tables and numerical
comparisons use all three seeds.} After every aggregation, global training,
validation and test CE, accuracy, balanced accuracy, AUROC and average precision
are logged, with runtime, optimizer steps and processed examples.

\textbf{When do we stop, and which model do we report?} A shared validation-CE plateau
scheduler halves LR after 11 bad rounds (PyTorch patience 10), with absolute
improvement tolerance 0.001, cooldown five, and floor initial LR/100.
Let $s_r$ be trailing-ten-round mean training CE. From round 50, each window of
20 smoothed values must have range $\leq\max(5\times10^{-4},0.005d)$ and absolute
OLS slope times 19 $\leq\max(10^{-4},0.001d)$, where
$d=\max(\operatorname{median}(s),10^{-12})$. Require five consecutive passing
windows and one LR throughout their 33-round raw-loss support. Absolute floors avoid excessive precision near zero; persistence and a fixed LR
exclude transient plateaus. We also record the original relative-only bounds.

\textbf{A run stops at loss stability, validation patience, or 300 total rounds,
including resumed rounds.} Predictive patience is 150 rounds without
cumulative validation-CE improvement $>10^{-4}$, eligible from round 80.
The lowest raw validation CE checkpoint (earliest tie) is restored
\cite{modelselection}. Its prediction threshold maximizes validation balanced
accuracy (highest threshold breaks ties). \textbf{Test metrics never control training, stopping, checkpoint selection or
thresholds.} \textbf{The restored checkpoint may precede the stable endpoint.} Neither loss
stability nor early stopping proves a stationary or global optimum. The 12 confirmed baseline training caches are reused unchanged.

\begin{table}[ht]\centering\scriptsize
\caption{Baseline selected-checkpoint results: mean $\pm$ sample SD over three
seeds. Stop rounds are ranges. BA and recall use validation-selected thresholds;
AUROC and CE are threshold-independent. Complete per-round metrics are in CSVs.}
\begin{tabular}{rrlllll}\toprule
$K$ & Stop rounds & Train CE & Test CE & Test AUROC & Test BA & Recall \\
\midrule
@@PERFORMANCE@@
\bottomrule\end{tabular}\end{table}
\begin{figure}[ht]\centering
\includegraphics[width=.96\linewidth]{training.png}
\caption{Baseline trajectories for \textbf{seed 684 only}. Colors distinguish
client counts; dots mark validation-selected checkpoints. Rounds 10 onward and restricted CE
axes resolve the plateau; complete histories are in CSV. No extrapolation after stopping.}
\end{figure}

\begin{samepage}
\textbf{Did every run converge?} @@STOPPING@@
\textbf{Loss stability does not establish good generalization.}\par
\end{samepage}

\begin{table}[ht]\centering\small
\caption{Paired-seed mean changes in selected training CE / test AUROC from the
lowest to highest parameter, with other parameters at baseline. Stop and
selected rounds differ across runs; negative CE indicates better fitting.}
\begin{tabular}{rrrr}\toprule
$K$ & Batch: 16 to 64 & Epochs: 1 to 5 & Initial LR: 0.01 to 0.1 \\
\midrule
@@EFFECTS@@
\bottomrule\end{tabular}\end{table}
\textbf{How do FedAvg settings affect training?} @@EFFICIENCY@@
Times include serial clients, evaluation and intermediate checkpoint I/O;
final selection/write and attack reconstruction are excluded. Different CPUs
and execution phases limit timing comparisons; network latency is not simulated.

\textbf{Does better training loss mean better readmission prediction?} @@GENERALIZATION@@
Regularization, scheduling, training size and selection changed together; the
historical comparison cannot isolate one cause. \textbf{Validation reused during
protocol development may give optimistic estimates.}

\section{Task B: First-Round LOSS Attacks}
\textbf{Task B asks us to run LOSS after the first round on the global model and
every client model, using the entire dataset.}
The honest-but-curious server follows FedAvg while inspecting individual
returned weights and the aggregated model. For each labelled candidate $(x,y)$,
the score is $-\ell(f(x),y)$: larger scores suggest membership \cite{loss}.
Only round one is attacked, before any validation-driven scheduling decision.
\textbf{Global members are the 3,600 optimization records; global nonmembers are
the 900 validation and 4,500 test records.} A local model's members are its own
optimization partition. \textbf{Local nonmembers are every record outside that
partition}, including other clients' optimization records, validation and test.
Local member/nonmember counts are $3,600/K$ and $9,000-3,600/K$.
Every model scores all 9,000 records.

We report ROC AUC and the highest empirical TPR with FPR at most 1\%, 5\%, or
10\%, preserving tied-score groups rather than interpolating observations.
\textbf{Oracle-threshold balanced accuracy} is
$\max_t[\mathrm{TPR}(t)+1-\mathrm{FPR}(t)]/2$.
\textbf{Membership labels select this threshold retrospectively.} This descriptive quantity is not independently
calibrated attack performance and differs from the classifier's validation
threshold. Every client's results are retained. We average clients within each seed, then
report mean and sample SD over three seeds; clients are not independent replicates.
The audit verifies all 1,785 first-round models, scores and aggregation.

\begin{table}[ht]\centering\scriptsize
\caption{Baseline first-round attacks. Local rows average clients within each
seed; entries are seed mean $\pm$ sample SD. BA is oracle-threshold balanced accuracy.}
\begin{tabular}{rlccccc}\toprule
$K$ & Model & Oracle BA & AUC & TPR@1\% & TPR@5\% & TPR@10\% \\
\midrule
@@ATTACKS@@
\bottomrule\end{tabular}\end{table}

\textbf{Are local attacks stronger, and how do settings affect them?} @@LEAKAGE@@
\textbf{Our explanation is that aggregation dilutes individual influence, while
small local partitions and more local steps encourage record-specific fitting.}
This is a hypothesis, not a causal finding. Local and global nonmembers have different mixtures, so their
AUC difference cannot isolate aggregation. Oracle BA increases can also reflect
retrospective threshold selection with fewer members. \textbf{Near-chance first-round LOSS does not establish privacy against stronger
attacks}, later-round leakage, or inference about real generator seed patients.

\textbf{Which step needs protection? Sending individual client weights to the
curious server exposes the models used by the local attacks.}
Transport encryption alone does not hide those weights from the receiving
server. \textbf{Secure aggregation hides individual updates by revealing only their
aggregate}, subject to an
appropriate aggregation cohort and dropout/collusion assumptions
\cite{secureagg}; a one-client cohort provides no such protection.
\textbf{The aggregate model can still leak membership; differential privacy
addresses that residual risk.} A formal guarantee requires clipping, calibrated
noise and privacy accounting; noise
must protect updates before the curious server can inspect them, or be applied
inside a trusted aggregation boundary. \textbf{Neither defense is evaluated here} \cite{dp}.

\section{Task C: Practicality and Future of FL}
\paragraph{Why is FL still mostly used in research?}
\textbf{I think the main difficulty is making the extra effort worthwhile.}
Keeping records local is useful, but hospitals still need to agree on features,
coordinate training and deal with different patient populations. Our simple IID
experiment already showed that changing client settings changes training behavior;
a real deployment would add more variation and operational work \cite{survey}.
FL also needs privacy protections beyond keeping raw records local.
Gboard and other Google systems show that production use exists
\cite{gboard,parfait}, but their success does not mean the same setup will suit
our startup. Larger generative models would add communication and resource costs.

\paragraph{What could drive wider adoption?}
\textbf{I would expect adoption to grow when pooling valuable data is difficult
and partners can show a clear benefit from working together.} For hospitals,
that benefit would need to justify the cost of agreeing on a schema, maintaining
the system and evaluating it. Easier tools, convincing privacy guarantees and
examples with measured gains would help. More FL papers alone would not answer
those practical questions.

\paragraph{Would our startup use FL?}
\textbf{I would start with a small cross-hospital pilot, not deploy this model.}
FL is appealing if partners want to collaborate without pooling records.
However, our synthetic results are only an initial check: baseline recall of
56--63\% still misses many positive cases, and changing the threshold creates
false positives. I would want committed partners, agreed features, an appropriate
validation mechanism and privacy protections before proceeding. Our centrally
accessible synthetic validation set is a simulation assumption. We have not
shown that the model works on real, heterogeneous hospital data or that it helps
clinical decisions.

\paragraph{Is there a better alternative?}
\textbf{If pooling is permitted and practical, I would try controlled centralized
training first.} It is simpler to train, debug and compare, although access
controls and privacy protections still matter. If pooling is impossible, local
models avoid shared updates but give up cross-hospital learning. Trusted execution
environments are another option, with hardware and attestation assumptions
\cite{survey,parfait}. I would choose based on the partners' actual constraints,
rather than assume FL is always the better privacy solution.

\textbf{Reproducibility and limits.}
Code, frozen data/partitions, hashes, all 16,088 round records and 1,785 attack
rows accompany the submission; bulky artifacts remain under \texttt{/work}.
Audits and 46 tests verify model/metric consistency, stopping, aggregation,
membership and resume. \textbf{Three seeds, reused validation and synthetic IID
data limit generalization.} Neither defenses nor hospital heterogeneity are
tested; the attack assessment covers first-round LOSS only.

\clearpage
\begin{thebibliography}{10}
\bibitem{uci} J. Clore, K. Cios, J. DeShazo, and B. Strack.
Diabetes 130-US Hospitals for Years 1999--2008. UCI, 2014.
\url{https://doi.org/10.24432/C5230J}.
\bibitem{tabddpm} A. Kotelnikov et al. TabDDPM: Modelling Tabular Data with Diffusion Models.
ICML, 2023. \url{https://arxiv.org/abs/2209.15421}.
\bibitem{fedavg} H. B. McMahan et al. Communication-Efficient Learning of Deep Networks
from Decentralized Data. AISTATS, 2017. \url{https://arxiv.org/abs/1602.05629}.
\bibitem{secureagg} K. Bonawitz et al. Practical Secure Aggregation for Privacy-Preserving
Machine Learning. CCS, 2017. \url{https://eprint.iacr.org/2017/281}.
\bibitem{survey} P. Kairouz et al. Advances and Open Problems in Federated Learning.
Foundations and Trends in Machine Learning, 2021. \url{https://arxiv.org/abs/1912.04977}.
\bibitem{gboard} Google Research. Improving Gboard language models via private federated
analytics. April 19, 2024.
\url{https://research.google/blog/improving-gboard-language-models-via-private-federated-analytics/}.
\bibitem{parfait} Google Research. Parfait: Enabling private AI with research tools.
January 22, 2025. \url{https://research.google/blog/parfait-enabling-private-ai-with-research-tools/}.
\bibitem{modelselection} Scikit-learn developers. Cross-validation: evaluating estimator performance;
MLPClassifier (validation-based early stopping). User Guide and API reference, accessed October 2026.
\url{https://scikit-learn.org/stable/modules/cross_validation.html}.
\url{https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html}.
\bibitem{loss} S. Yeom et al. Privacy Risk in Machine Learning:
Analyzing the Connection to Overfitting. IEEE CSF, 2018.
\url{https://arxiv.org/abs/1709.01604}.
\bibitem{dp} M. Abadi et al. Deep Learning with Differential Privacy.
CCS, 2016. \url{https://arxiv.org/abs/1607.00133}.
\end{thebibliography}
\end{document}
"""

SLIDE_TEMPLATE = r"""\documentclass[aspectratio=169]{beamer}
\usepackage{graphicx}
\setbeamertemplate{navigation symbols}{}
\begin{document}
\begin{frame}{FedAvg and first-round membership inference}
\scriptsize \textbf{Team 08} --- Assignment lead: Chetan Kodand Reddy Madadi\\
Team members: Bharat Govil, Naveen Jarpla, Vishnu Fnu
\begin{columns}[T]
\begin{column}{.54\textwidth}
\begin{itemize}
\setlength{\itemsep}{1pt}
\item 9,000 synthetic encounters; Assignment 1 generation pipeline reused.
\item 50/50 split: 3,600 SGD + 900 validation / 4,500 test.
\item 28 FedAvg settings $\times$ 3 seeds; 300-round cap.
\item L2 + validation scheduler; best validation checkpoint restored.
\item \textbf{Central synthetic validation is a simulation assumption.}
\item \textbf{72 loss-stable; 9 early stops; 3 capped.}
\item Baseline AUROC 0.657--0.707; validation-threshold BA 0.606--0.638.
\item $K=1$ test CE: 1.275 $\rightarrow$ 0.323; moderate predictive utility.
\end{itemize}
\end{column}
\begin{column}{.43\textwidth}
\includegraphics[width=\linewidth]{figures/attack_auc.png}
\scriptsize \textbf{Plot: seed 684 only.} Listed metrics are three-seed means. All 84 runs and 1,785 attacks audited.
First-round LOSS remains approximately chance; this is not a privacy guarantee.
\end{column}
\end{columns}
\medskip\scriptsize
Local nonmembers: all records outside that client's optimization partition.\quad
Oracle BA chooses its threshold retrospectively.\\
Secure aggregation hides individual updates; differential privacy limits membership leakage.
\end{frame}
\end{document}
"""

def number(value):
    return f"{value:.4f}"


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return f"{values.mean():.4f} $\\pm$ {values.std(ddof=1):.4f}" if len(values) > 1 else number(values[0])


def render_original(destination, config, audit, statuses, attacks):
    from common import load_config, read_json, sha256_file
    destination = Path(destination)
    from analyze import baseline_mask, make_training_figure, training_diagnostics
    history_path = destination / "results" / "round_metrics.csv"
    histories = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    diagnostics = training_diagnostics(histories, config) if not histories.empty else pd.DataFrame()
    if not diagnostics.empty:
        diagnostics.to_csv(destination / "results" / "training_diagnostics.csv", index=False)
        make_training_figure(destination, config, histories)
    done = statuses.loc[statuses.state.isin(["converged", "capped_unconverged"])] if not statuses.empty else statuses
    baseline = done.copy()
    for key, value in config["baseline"].items():
        if not baseline.empty:
            baseline = baseline.loc[baseline[key] == value]
    complete = audit["complete"]
    coverage = (f"{audit['finalized_runs']}/84 runs finalized; "
                f"{audit['verified_model_evaluations']:,}/1,785 first-round models verified.")
    counts = audit["counts"]
    convergence_text = (f"{counts.get('converged', 0)} runs met the convergence criterion and "
                        f"{counts.get('capped_unconverged', 0)} reached the 300-round cap without convergence.")
    performance_rows = []
    for clients, group in baseline.groupby("clients") if not baseline.empty else []:
        performance_rows.append(f"{clients} & {len(group)} & {int(group['rounds'].min())}--{int(group['rounds'].max())} & "
                                f"{mean_std(group.test_accuracy)} & {mean_std(group.test_balanced_accuracy)} & "
                                f"{mean_std(group.test_auroc)} & "
                                f"{mean_std(group.test_average_precision)} \\\\")
    performance_table = "\n".join(performance_rows) or r"\multicolumn{7}{c}{No finalized training results available.} \\"
    attack_rows, effects_rows = [], []
    baseline_attacks = attacks.copy()
    if not attacks.empty:
        for key, value in config["baseline"].items():
            baseline_attacks = baseline_attacks.loc[baseline_attacks[key] == value]
        for (clients, model_type), group in baseline_attacks.groupby(["clients", "model_type"]):
            seed_means = group.groupby("seed")[["oracle_threshold_balanced_accuracy", "auc",
                                               "tpr_at_1pct_fpr", "tpr_at_5pct_fpr", "tpr_at_10pct_fpr"]].mean()
            attack_rows.append(f"{clients} & {model_type.capitalize()} & " + " & ".join(
                mean_std(seed_means[key]) for key in seed_means) + r" \\")
    attack_table = "\n".join(attack_rows) or r"\multicolumn{7}{c}{No verified attack results available.} \\"
    for clients in config["client_counts"]:
        deltas = []
        for parameter in ["batch_size", "local_epochs", "learning_rate"]:
            subset = done.loc[done.clients == clients].copy() if not done.empty else done
            for key, value in config["baseline"].items():
                if key != parameter and not subset.empty:
                    subset = subset.loc[subset[key] == value]
            if subset.empty:
                deltas.append("--")
            else:
                lowest, highest = min(config["variations"][parameter]), max(config["variations"][parameter])
                low = subset.loc[subset[parameter] == lowest, ["seed", "train_cross_entropy", "test_auroc"]]
                high = subset.loc[subset[parameter] == highest, ["seed", "train_cross_entropy", "test_auroc"]]
                paired = high.merge(low, on="seed", suffixes=("_high", "_low"))
                deltas.append(f"{(paired.train_cross_entropy_high - paired.train_cross_entropy_low).mean():+.4f} / "
                              f"{(paired.test_auroc_high - paired.test_auroc_low).mean():+.4f}" if len(paired) else "--")
        effects_rows.append(f"{clients} & " + " & ".join(deltas) + r" \\")
    effects_table = "\n".join(effects_rows)
    if not baseline_attacks.empty:
        means = baseline_attacks.groupby(["clients", "seed", "model_type"]).auc.mean().unstack("model_type")
        differences = (means.local - means['global']).groupby("clients").mean()
        leakage_text = "Observed baseline local-minus-global AUC differences (client means, then seed means) are " + "; ".join(
            f"$K={int(k)}$: {v:+.4f}" for k, v in differences.items()) + ". "
        global_auc = baseline_attacks.loc[baseline_attacks.model_type == "global"].groupby("clients").auc.mean()
        headline = f"Baseline global LOSS AUC (seed means): {global_auc.min():.3f}--{global_auc.max():.3f}."
    else:
        leakage_text = "No verified results are available yet; no direction or magnitude of leakage is claimed. "
        headline = "Experimental results pending; no leakage conclusion yet."
    if not baseline.empty:
        utility = baseline.groupby("clients").test_auroc.mean()
        headline_utility = f"Baseline test AUROC (seed means): {utility.min():.3f}--{utility.max():.3f}."
    else:
        headline_utility = "Training results pending."
    status_note = "Complete artifact coverage." if complete else "INCOMPLETE EXPERIMENT COVERAGE: results are provisional."
    prevalence_text = ""
    training_observations = "No finalized training observations are available."
    convergence_observations = "No finalized loss-history diagnostics are available."
    efficiency_observations = "No finalized computation measurements are available."
    parameter_observations = "No verified parameter-effect observations are available."
    pilot_observations = ""
    generalization_observations = ""
    generalization_headline = ""
    pilot_path = destination / "results" / "convergence_pilot_summary.csv"
    if pilot_path.exists():
        proof = read_json(destination / "results" / "convergence_pilot_validation.json")
        if not proof['passed'] or sha256_file(pilot_path) != proof['summary_sha256']:
            raise ValueError("Unverified convergence-pilot summary")
        pilots = pd.read_csv(pilot_path)
        expected = {(k, policy, 684) for k in config['client_counts']
                    for policy in ['tail_scaled_sgd', 'step_decay_sgd']}
        if len(pilots) != 8 or set(zip(pilots.clients, pilots.policy, pilots.seed)) != expected:
            raise ValueError("Incomplete convergence pilots")
        scheduled = pilots.loc[pilots.policy == 'step_decay_sgd'].set_index('clients')
        passing = scheduled.loc[[10, 20, 50]]
        tail = pilots.loc[(pilots.policy == 'tail_scaled_sgd') & (pilots.clients == 10)].iloc[0]
        if (passing.state.ne('converged').any() or scheduled.loc[1, 'state'] != 'capped_unconverged'
                or pilots.loc[pilots.policy == 'tail_scaled_sgd', 'state'].ne('capped_unconverged').any()):
            raise ValueError("Update the convergence-pilot interpretation")
        pilot_observations = (
            r"\textbf{Exploratory follow-up (seed 684 only).} Eight pilots retain the original campaign. "
            "LRs 0.05, 0.005, and 0.0005 in successive 100-round blocks make $K=10/20/50$ pass "
            f"the same rule at rounds {'/'.join(str(int(v)) for v in passing['rounds'])}, "
            f"but test AUROC is {'/'.join(f'{v:.4f}' for v in passing.pilot_test_auroc)} "
            f"versus {'/'.join(f'{v:.4f}' for v in passing.control_test_auroc)} for the original controls. "
            "All three predict only negatives; full-training regularized gradient norms "
            f"{passing.pilot_regularized_gradient_l2.min():.3f}--{passing.pilot_regularized_gradient_l2.max():.3f} "
            "do not establish stationarity. $K=1$ stays capped. Partial-batch LR scaling by actual size/32 "
            "passes no run, although $K=10$ test loss improves "
            f"{tail.control_test_cross_entropy:.4f} to {tail.pilot_test_cross_entropy:.4f}. "
            "These results do not justify replacing the sweep.")
    protocol_path = destination / "results" / "task_a_protocol_summary.csv"
    if protocol_path.exists():
        proof = read_json(destination / "results" / "task_a_protocol_validation.json")
        exports = {'summary.csv': 'task_a_protocol_summary.csv',
                   'protocol_comparison.csv': 'task_a_protocol_comparison.csv',
                   'tail_confirmation.csv': 'task_a_tail_confirmation.csv',
                   'round_metrics.csv': 'task_a_protocol_history.csv'}
        if (not proof['passed'] or proof['audited_trials'] != 16 or proof['failed_trials']
                or set(proof['artifact_sha256']) != set(exports)
                or sha256_file(destination / 'task_a_protocol.json') != proof['protocol_sha256']):
            raise ValueError("Incomplete or incompatible Task A protocol validation")
        for original, exported in exports.items():
            if sha256_file(destination / 'results' / exported) != proof['artifact_sha256'][original]:
                raise ValueError("Corrupt Task A protocol export")
        revised = pd.read_csv(protocol_path)
        comparisons = pd.read_csv(destination / 'results' / exports['protocol_comparison.csv'])
        confirmations = pd.read_csv(destination / 'results' / exports['tail_confirmation.csv'])
        active = revised.loc[revised.policy == 'tail_plateau']
        if (len(revised) != 16 or len(active) != 12 or len(comparisons) != 16
                or len(confirmations) != 6 or
                set(zip(active.clients, active.seed)) != {(k,s) for k in config['client_counts'] for s in config['seeds']}):
            raise ValueError("Incomplete paired Task A follow-up")
        small = active.loc[active.clients == 1].set_index('seed').loc[config['seeds']]
        larger = active.loc[active.clients != 1]
        if (small.state.ne('loss_stable').any() or larger.state.ne('capped_unconverged').any()
                or any(run['lr_reductions'] for run in proof['runs'])):
            raise ValueError("Update Task A follow-up interpretation")
        ten = confirmations.loc[confirmations.clients == 10]
        twenty = confirmations.loc[confirmations.clients == 20]
        for group in [ten, twenty]:
            if len(group) != 3 or set(group.seed) != set(config['seeds']):
                raise ValueError("Incomplete three-seed tail confirmation")
            if not (group.trial_last50_training_loss_range < group.control_last50_training_loss_range).all():
                raise ValueError("Update tail-update stability finding")
        if not (ten.trial_test_auroc > ten.control_test_auroc).all() or not (
                twenty.trial_test_auroc < twenty.control_test_auroc).all():
            raise ValueError("Update tail-update generalization finding")
        proto = read_json(destination / 'task_a_protocol.json')
        protocol_history = pd.read_csv(destination / 'results' / exports['round_metrics.csv'])
        late_decreases = []
        for clients in [10, 20, 50]:
            group = protocol_history.loc[(protocol_history.policy == 'tail_plateau') &
                                         (protocol_history.clients == clients)]
            changes = []
            for seed in config['seeds']:
                curve = group.loc[group.seed == seed].sort_values('round')
                if curve['round'].tolist() != list(range(1, 301)):
                    raise ValueError("Incomplete baseline follow-up history")
                loss = curve.train_cross_entropy.to_numpy()
                changes.append(float(loss[200:250].mean() - loss[250:300].mean()))
            if min(changes) <= 0:
                raise ValueError("Update the sustained-training-improvement interpretation")
            late_decreases.append(float(np.mean(changes)))
        pilot_observations = (
            r"\textbf{Targeted protocol follow-up.} "
            "Calendar-decay pilots passed the old rule for $K=10/20/50$ but reduced AUROC "
            "and predicted only negatives. Sixteen baseline trials scale partial-batch LR by "
            "actual size/$B$; 12 also use training-loss plateau decay (factor 0.5, patience 20). "
            "No LR reductions occurred; no scheduling benefit is claimed. Original runs remain unchanged.\n\n"
            "Prospective range/trend bounds are "
            r"$\max(5\times10^{-4},0.005m)$ and $\max(10^{-4},0.001m)$, "
            f"with median $m$, {proto['consecutive_windows']} consecutive passing windows, "
            f"{proto['fixed_lr_rounds']} fixed-LR rounds, and cap 300. Tolerances were frozen before training. "
            f"$K=1$ is loss-stable at rounds {'/'.join(str(int(v)) for v in small['rounds'])} "
            "for seeds 684/685/686. Mean train/test loss is "
            f"{small.train_cross_entropy.mean():.4f}/{small.test_cross_entropy.mean():.4f}; "
            f"regularized full-training gradient norms ({1e3*small.regularized_gradient_l2.min():.1f}--"
            f"{1e3*small.regularized_gradient_l2.max():.1f})" + r"$\times10^{-3}$ "
            "are diagnostic. Overfitting persists; practical stability does not establish stationarity. "
            f"The other {proof['capped_unconverged']} trials remain capped.\n\n"
            "Three-seed constant-tail confirmations (reusing two seed-684 models) "
            "reduce mean last-50-round loss ranges: $K=10$, "
            f"{ten.control_last50_training_loss_range.mean():.4f} to {ten.trial_last50_training_loss_range.mean():.4f}"
            f"; $K=20$, {twenty.control_last50_training_loss_range.mean():.4f} to "
            f"{twenty.trial_last50_training_loss_range.mean():.4f}. "
            f"$K=10$ test AUROC improves {ten.control_test_auroc.mean():.4f} to {ten.trial_test_auroc.mean():.4f} "
            f"in all seeds; $K=20$ decreases {twenty.control_test_auroc.mean():.4f} to "
            f"{twenty.trial_test_auroc.mean():.4f} in all seeds. "
            "Stability does not guarantee discrimination. Adaptive-baseline mean training-loss decreases "
            "between rounds 201--250 and 251--300 are "
            f"{'/'.join(f'{v:.4f}' for v in late_decreases)} at $K=10/20/50$, each positive in all seeds. "
            "Bounds are not noise-calibrated; failure alone does not prove divergence. "
            "Full-sweep convergence remains unresolved.")
    generalization_path = destination / 'results/generalization_summary.csv'
    if generalization_path.exists():
        from generalization_followup import PROTOCOL
        validation = read_json(destination / 'results/generalization_validation.json')
        exports = {'summary.csv': 'generalization_summary.csv',
                   'round_metrics.csv': 'generalization_history.csv'}
        if (not validation['passed'] or validation['audited_trials'] != 12 or validation['failed_trials']
                or validation['protocol'] != PROTOCOL or validation['convergence_claim']
                or validation['selected_and_last_models_rescored'] != 24):
            raise ValueError("Incomplete or incompatible generalization audit")
        for original, exported in exports.items():
            if sha256_file(destination / 'results' / exported) != validation['artifact_sha256'][original]:
                raise ValueError("Corrupt generalization export")
        partition_path = destination / 'data/generalization_partitions.json'
        if sha256_file(partition_path) != validation['partition_manifest_sha256']:
            raise ValueError("Corrupt generalization partitions")
        partitions = read_json(partition_path)
        if (partitions['protocol'] != PROTOCOL or partitions['dataset_sha256'] !=
                read_json(destination / 'data/dataset_manifest.json')['sha256']):
            raise ValueError("Incompatible generalization partition definition")
        predictive = pd.read_csv(generalization_path)
        predictive_history = pd.read_csv(destination / 'results/generalization_history.csv')
        if (len(predictive) != 12 or predictive[['clients','seed']].duplicated().any() or
                set(zip(predictive.clients,predictive.seed)) !=
                {(k,s) for k in config['client_counts'] for s in config['seeds']} or
                len(predictive_history) != validation['history_rows'] or
                len(predictive_history) != predictive['rounds'].sum() or
                not predictive.training_records.eq(3600).all() or
                not predictive.validation_records.eq(900).all() or not predictive.test_records.eq(4500).all()):
            raise ValueError("Incomplete generalization coverage")
        for _, row in predictive.iterrows():
            control = baseline.loc[(baseline.clients == row.clients) & (baseline.seed == row.seed)]
            if len(control) != 1 or not 1 <= row.selected_round <= row['rounds'] <= 300:
                raise ValueError("Invalid paired generalization result")
            for key in ['train_cross_entropy','test_cross_entropy','test_auroc']:
                if not np.isclose(row['original_'+key],control.iloc[0][key],rtol=1e-10,atol=1e-12):
                    raise ValueError("Generalization original-control mismatch")
            curve = predictive_history.loc[(predictive_history.clients == row.clients) &
                                           (predictive_history.seed == row.seed)].sort_values('round')
            if (curve['round'].tolist() != list(range(1,int(row['rounds'])+1)) or
                    int(curve.loc[curve.validation_cross_entropy.idxmin(),'round']) != row.selected_round):
                raise ValueError("Generalization model is not the earliest validation minimum")
        early_count = int(predictive.state.eq('validation_early_stopped').sum())
        capped_count = int(predictive.state.eq('capped').sum())
        all_negative = int(predictive.test_positive_predictions.eq(0).sum())
        if early_count+capped_count != 12:
            raise ValueError("Unsuccessful generalization trial not represented")
        central_predictive = predictive.loc[predictive.clients == 1].set_index('seed').loc[config['seeds']]
        predictive_means = predictive.groupby('clients').mean(numeric_only=True)
        sensitivity_text = '/'.join(f'{100*predictive_means.loc[k,"test_sensitivity"]:.2f}'
                                    for k in config['client_counts'])
        table_rows = []
        for clients, group in predictive.groupby('clients'):
            table_rows.append(f"{clients} & {int(group['rounds'].min())}--{int(group['rounds'].max())} & "
                              f"{int(group.selected_round.min())}--{int(group.selected_round.max())} & "
                              f"{mean_std(group.test_cross_entropy)} & {mean_std(group.test_auroc)} " + r"\\")
        generalization_observations = (
            r"\textbf{Predictive stopping follow-up.} "
            "Twelve baselines retain the 4,500-record test half, reserve 900 stratified validation "
            "records, and optimize 3,600 records in equal IID client partitions, reusing original SGD "
            "and public preprocessing. Validation CE controls stopping: "
            f"minimum {PROTOCOL['min_rounds']} rounds, patience {PROTOCOL['patience']}, "
            r"improvement threshold $10^{-4}$, cap 300. "
            "Restore the earliest validation minimum; cumulative improvements above the threshold reset patience. "
            f"{early_count} runs stop on patience; {capped_count} cap. These are not convergence claims.\n\n"
            r"\begin{table}[H]\centering\scriptsize" + "\n"
            r"\caption{Validation-selected checkpoints; test metrics are three-seed mean $\pm$ sample SD. "
            r"Rounds show ranges.}" + "\n"
            r"\begin{tabular}{rrrll}\toprule $K$ & Stop rounds & Selected rounds & Test CE & Test AUROC \\" + "\n"
            r"\midrule" + "\n" + '\n'.join(table_rows) + "\n"
            r"\bottomrule\end{tabular}\end{table}" + "\n\n"
            f"$K=1$ mean train/test CE is {central_predictive.train_cross_entropy.mean():.4f}/"
            f"{central_predictive.test_cross_entropy.mean():.4f}, versus "
            f"{central_predictive.original_train_cross_entropy.mean():.4f}/"
            f"{central_predictive.original_test_cross_entropy.mean():.4f} originally. "
            "Mean test AUROC falls for every $K$. "
            f"{all_negative}/12 selected models predict only negatives; mean sensitivity for $K=1/10/20/50$ is "
            f"{sensitivity_text}" +
            r"\%. Reduced $K=1$ overfitting does not establish readmission utility. "
            "Training size and selection both change; differences cannot isolate stopping alone. "
            "Task B remains the original campaign.")
        generalization_headline = (
            r"\item Validation: $K=1$ test CE "
            f"{central_predictive.original_test_cross_entropy.mean():.3f} to "
            f"{central_predictive.test_cross_entropy.mean():.3f}; AUROC "
            f"{central_predictive.original_test_auroc.mean():.3f} to {central_predictive.test_auroc.mean():.3f}. "
            f"{all_negative}/12 predict only negatives.")
        if protocol_path.exists():
            pilot_observations = (
                r"\textbf{Optimization follow-ups.} "
                "Eight exploratory pilots and sixteen revised-stopping trials did not resolve $K=1$ overfitting. "
                "$K=1$ meets the revised loss-stability rule at 221/225/217; "
                f"mean test CE remains {small.test_cross_entropy.mean():.4f}. Other 13 trials cap at 300. "
                "Criteria and complete pilot results are archived; stability does not establish generalization.")
    manifest_path = destination / "data" / "dataset_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        prevalence_text = (f"The regenerated positive-label prevalence is "
                           f"{manifest['positive_count'] / manifest['n_records']:.4f}. ")

    if complete and not baseline.empty:
        if len(diagnostics) != audit['finalized_runs']:
            raise ValueError("Report is missing finalized loss-history diagnostics")
        central = baseline.loc[baseline.clients == 1]
        many = baseline.loc[baseline.clients == 50]
        db = diagnostics.loc[baseline_mask(diagnostics, config)]
        dcentral, dmany = db.loc[db.clients == 1], db.loc[db.clients == 50]
        noisy = db.loc[db.clients == 10]
        settings = config['convergence']
        if diagnostics.first_criterion_round.notna().any() or done.state.ne('capped_unconverged').any():
            raise ValueError("Update report interpretation: the recorded campaign is no longer uniformly capped")
        range_pass = int((diagnostics.smoothed_relative_range <= settings['relative_range']).sum())
        trend_pass = int((diagnostics.smoothed_relative_trend <= settings['relative_trend']).sum())
        convergence_observations = (
            f"No earlier crossing occurs. At round 300, "
            f"{range_pass}/84 pass the range bound, {trend_pass}/84 the trend bound, and none both. "
            f"$K=1$ has fitted window change {1e5 * dcentral.smoothed_signed_change.abs().mean():.1f}"
            r"$\times10^{-5}$, yet a "
            f"{100 * dcentral.smoothed_relative_trend.mean():.1f}\\% relative change: the rule is conservative. "
            f"$K=50$ trend is "
            f"{100 * dmany.smoothed_relative_trend.min():.2f}--"
            f"{100 * dmany.smoothed_relative_trend.max():.2f}\\%, above 0.10\\%; improvement continues. "
            f"$K=10$ oscillates: last-50-round loss ranges "
            f"{noisy.last_50_training_loss_range.min():.3f}--{noisy.last_50_training_loss_range.max():.3f}. "
            "Losses remain finite; original capped labels remain.")
        base_history = histories.loc[baseline_mask(histories, config) & (histories.clients == 1)]
        mean_curve = base_history.groupby('round')[['train_cross_entropy', 'test_cross_entropy']].mean()
        min_round = int(mean_curve.test_cross_entropy.idxmin())
        min_loss = float(mean_curve.loc[min_round, 'test_cross_entropy'])
        training_observations = (
            f"At baseline, mean train/test cross-entropy is {central.train_cross_entropy.mean():.4f}/"
            f"{central.test_cross_entropy.mean():.4f} for $K=1$, compared with "
            f"{many.train_cross_entropy.mean():.4f}/{many.test_cross_entropy.mean():.4f} for $K=50$. "
            f"For $K=1$, seed-mean test loss reaches its observed minimum {min_loss:.4f} at round "
            f"{min_round}, then rises to {central.test_cross_entropy.mean():.4f} while training loss falls: "
            "continued fitting overfits. This retrospective test minimum never selects a model. "
            f"At $K=50$, mean accuracy {many.test_accuracy.mean():.4f} coexists with balanced accuracy "
            f"{many.test_balanced_accuracy.mean():.4f}; "
            "all three models predict only negatives at the default threshold. "
            "Sensitivity is zero despite above-chance AUROC. "
            "Fixed-round outcomes mix slow learning, oscillation, and overfitting; "
            "they neither establish converged optima nor equalize work.\n\n"
            "Predictive early stopping reserves separate validation data and restores its best "
            r"checkpoint \cite{modelselection}. "
            f"The original study has no validation set; test round {min_round} cannot select a model.")
        def setting_mean(clients, parameter, value):
            subset = done.loc[done.clients == clients].copy()
            for key, default in config['baseline'].items():
                subset = subset.loc[subset[key] == (value if key == parameter else default)]
            if len(subset) != len(config['seeds']):
                raise ValueError("Incomplete paired parameter comparison")
            return subset.mean(numeric_only=True)
        b16, b64 = setting_mean(50, 'batch_size', 16), setting_mean(50, 'batch_size', 64)
        e1, e3, e5 = [setting_mean(50, 'local_epochs', e) for e in [1, 3, 5]]
        lr_low, lr_high = setting_mean(50, 'learning_rate', .01), setting_mean(50, 'learning_rate', .1)
        ten_low, ten_high = setting_mean(10, 'learning_rate', .01), setting_mean(10, 'learning_rate', .1)
        efficiency_observations = (
            r"\textbf{Work and parameter effects.} "
            f"All baseline runs process {many.cumulative_processed_examples.mean()/1e6:.2f} million examples "
            f"in 300 rounds, but a client performs 141 updates/round at $K=1$ versus 3 at $K=50$. "
            "Averaging many short local paths therefore differs from one long SGD path. "
            f"Total updates are {central.cumulative_optimizer_steps.mean()/1000:.1f}k versus "
            f"{many.cumulative_optimizer_steps.mean()/1000:.1f}k, and mean recorded times "
            f"{central.elapsed_seconds.mean():.0f} versus {many.elapsed_seconds.mean():.0f} s.\n\n"
            f"At $K=50$, batches 16 to 64 reduce updates "
            f"{b16.cumulative_optimizer_steps/1000:.0f}k to {b64.cumulative_optimizer_steps/1000:.0f}k "
            f"and time {b16.elapsed_seconds:.0f} to {b64.elapsed_seconds:.0f} s, but raise train loss "
            f"{b16.train_cross_entropy:.4f} to {b64.train_cross_entropy:.4f} and lower test AUROC "
            f"{b16.test_auroc:.4f} to {b64.test_auroc:.4f}. "
            f"Epochs 1/3/5 cost {e1.cumulative_optimizer_steps/1000:.0f}/"
            f"{e3.cumulative_optimizer_steps/1000:.0f}/{e5.cumulative_optimizer_steps/1000:.0f}k updates, "
            f"{e1.cumulative_processed_examples/1e6:.2f}/{e3.cumulative_processed_examples/1e6:.2f}/"
            f"{e5.cumulative_processed_examples/1e6:.2f} million examples, and "
            f"{e1.elapsed_seconds:.0f}/{e3.elapsed_seconds:.0f}/{e5.elapsed_seconds:.0f} s. "
            f"Train loss falls {e1.train_cross_entropy:.4f}/{e3.train_cross_entropy:.4f}/{e5.train_cross_entropy:.4f}, "
            f"while test loss is {e1.test_cross_entropy:.4f}/{e3.test_cross_entropy:.4f}/{e5.test_cross_entropy:.4f} "
            f"and AUROC {e1.test_auroc:.4f}/{e3.test_auroc:.4f}/{e5.test_auroc:.4f}: "
            "extra local work eventually worsens generalization.\n\n"
            f"LR 0.01 to 0.1 keeps work counts fixed but lowers train loss "
            f"{lr_low.train_cross_entropy:.4f} to {lr_high.train_cross_entropy:.4f} and raises test AUROC "
            f"{lr_low.test_auroc:.4f} to {lr_high.test_auroc:.4f}; "
            "learning faster still does not satisfy the stability rule. "
            f"At $K=10$, the same LR increase lowers train loss {ten_low.train_cross_entropy:.4f} to "
            f"{ten_high.train_cross_entropy:.4f} but raises test loss {ten_low.test_cross_entropy:.4f} to "
            f"{ten_high.test_cross_entropy:.4f}, so improved fitting does not ensure better generalization. "
            "Recorded times include serial clients, evaluation, and first-round attacks; "
            "exclude the final metric/checkpoint write; and omit network/parallel latency.")
    if complete and not attacks.empty:
        attack_keys = ["clients", "batch_size", "local_epochs", "learning_rate", "model_type"]
        per_seed = attacks.groupby(attack_keys + ["seed"])[["auc", "tpr_at_1pct_fpr"]].mean()
        means = per_seed.groupby(level=attack_keys).mean()
        base = config['baseline']
        def local_mean(parameter, value):
            setting = dict(base)
            setting[parameter] = value
            return means.loc[(50, setting['batch_size'], setting['local_epochs'], setting['learning_rate'], 'local')]
        one, five = local_mean('local_epochs', 1), local_mean('local_epochs', 5)
        batch_low, batch_high = local_mean('batch_size', 16), local_mean('batch_size', 64)
        lr_low, lr_high = local_mean('learning_rate', .01), local_mean('learning_rate', .1)
        global_means = means.loc[means.index.get_level_values('model_type') == 'global'].auc
        parameter_observations = (
            f"For $K=50$, increasing local epochs from 1 to 5 raises local mean AUC from "
            f"{one.auc:.4f} to {five.auc:.4f}, and TPR at 1\\% FPR from "
            f"{100 * one.tpr_at_1pct_fpr:.2f}\\% to {100 * five.tpr_at_1pct_fpr:.2f}\\%. "
            f"Increasing batch size from 16 to 64 lowers local mean AUC from {batch_low.auc:.4f} to "
            f"{batch_high.auc:.4f}; increasing learning rate from 0.01 to 0.1 raises it from "
            f"{lr_low.auc:.4f} to {lr_high.auc:.4f}. "
            f"Global configuration-mean AUCs remain {global_means.min():.4f}--{global_means.max():.4f} "
            "across the entire study. The observed local advantage remains weak. "
            "Oracle BA also benefits from selecting the best threshold and can be more optimistic with "
            "fewer members; its increase alone does not establish a strong membership attack.")

    source = r"""\documentclass[11pt]{article}
\usepackage[margin=0.8in]{geometry}
\usepackage{amsmath,booktabs,graphicx,microtype,hyperref,float}
\hypersetup{colorlinks=true,urlcolor=blue,citecolor=blue}
\graphicspath{{figures/}}
\setlength{\parskip}{3pt}
\widowpenalty=10000
\clubpenalty=10000
\title{Assignment 2: Federated Learning and Membership Inference}
\author{Team 08 \\ Chetan Kodand Reddy Madadi \\ Main author and project lead}
\date{Fall 2026}
\begin{document}
\maketitle
\begin{abstract}
We evaluate full-participation FedAvg for a healthcare readmission classifier and
record-level membership inference by an honest-but-curious server. A new run of
Assignment 1's mixed TabDDPM generates 9,000 synthetic encounters. We compare 28
controlled configurations with three seeds, and attack every first-round client
model and aggregated global model using negative cross-entropy.
@@COVERAGE@@ In the original campaign, @@CONVERGENCE@@ \textbf{@@STATUS@@}
\end{abstract}

\section{Task A: Data, FedAvg, and Experimental Design}
Our startup use case is predicting 30-day readmission at discharge across hospitals.
The public UCI Diabetes 130-US Hospitals data are licensed CC BY 4.0
\cite{uci}. We preserve Assignment 1's cleaning, medication-derived fields,
patient-disjoint source partitions, and class-conditional mixed TabDDPM
\cite{tabddpm}. The generator uses 20,000 steps, 200 diffusion steps, batch size
1,024, width 512, four residual blocks, AdamW at $2\times10^{-4}$, and EMA 0.999.
These are newly generated records, not the original Assignment 1 dataset.
Duplicate, exact seed-match, medication-consistency, and excluded-disposition
audits accompany the frozen CSV and SHA-256 manifest.
@@PREVALENCE@@ Generated outcomes are synthetic labels, not verified clinical outcomes;
synthetic data provide no formal privacy guarantee. The membership question concerns
synthetic classifier records, not real patients in generator training.

A shared preprocessor is fitted only on the public generator seed records:
median imputation and standardization for numerical fields, and fixed-schema
one-hot encoding for categories. No synthetic test record fits any transformation.
This public-background assumption aligns client feature coordinates without
collecting FL clients' raw records. Record and patient identifiers and the raw
outcome field are excluded from model inputs.

For each seed (684, 685, 686), a label-stratified 50/50 split supplies 4,500
training and 4,500 test records. A random permutation partitions training records
equally and IID among $K\in\{1,10,20,50\}$ clients. Natural finite-sample label
variation is retained; partitions remain fixed across parameter comparisons.
The MLP has ReLU hidden layers of widths 128 and 64 and two output logits.
Every round, all clients copy the same global weights and perform local SGD
without momentum, with unweighted cross-entropy and weight decay $10^{-5}$.
The server updates
\[
W_{r+1}=\sum_{i=1}^{K}\frac{n_i}{\sum_j n_j}W_{i,r+1},
\]
which is the arithmetic mean here because all clients have equal sizes
\cite{fedavg}. $K=1$ is centralized SGD with the same round/epoch grouping.

The baseline uses batch size 32, one local epoch, and learning rate 0.05.
One parameter at a time varies over batches $\{16,32,64\}$, epochs $\{1,3,5\}$,
or learning rates $\{0.01,0.05,0.1\}$. Deduplication gives seven settings per
client count, 28 configurations, and 84 seeded runs. Batches are shuffled and
never dropped. Global train/test loss, accuracy, balanced accuracy, AUROC, and
average precision are recorded after every round, alongside time, steps, and
processed examples. This design measures individual effects, not interactions.

Training stops by 300 rounds. Let $s_r$ be the trailing 10-round mean of global
training cross-entropy. Starting at round 50, the latest 20 values must have
range at most $0.005d$ and fitted absolute linear trend across the window at
most $0.001d$, where $d=\max(\operatorname{median}(s),10^{-12})$.
Test metrics never determine stopping. Capped, unconverged and failed runs are
explicit outcomes; checkpoint resume preserves the total-round limit.
This tests loss stability, not gradient stationarity.

\begin{table}[ht]
\centering\scriptsize
\caption{Baseline final test performance; mean $\pm$ sample standard deviation across available seeds.
Rounds show the observed minimum--maximum. All configurations are in the accompanying CSVs.}
\begin{tabular}{rrrllll}
\toprule $K$ & Seeds & Rounds & Accuracy & Balanced acc. & AUROC & Avg. precision \\
\midrule
@@PERFORMANCE@@
\bottomrule\end{tabular}
\end{table}
\IfFileExists{figures/training.png}{\begin{figure}[ht]\centering
\includegraphics[width=.96\linewidth]{training.png}
\caption{Baseline trajectories: training loss (log scale), test loss, and test AUROC.
Solid curves average seeds; faint loss curves show individual runs.}\end{figure}}{}

\textbf{Loss-stability diagnosis.} @@CONVERGENCE_OBSERVATIONS@@

@@PILOT_OBSERVATIONS@@

\begin{table}[ht]
\centering\small
\caption{Paired-seed mean changes in final training cross-entropy / test AUROC,
from the lowest to highest parameter value, with other parameters at baseline.
Negative training-loss changes indicate better fitting, not necessarily better generalization.}
\begin{tabular}{rrrr}\toprule $K$ & Batch: 16 to 64 & Epochs: 1 to 5 & LR: 0.01 to 0.1 \\
\midrule
@@EFFECTS@@
\bottomrule\end{tabular}\end{table}
@@EFFICIENCY_OBSERVATIONS@@

\textbf{Generalization and imbalance.}
@@TRAINING_OBSERVATIONS@@

@@GENERALIZATION_FOLLOWUP@@

\section{Task B: First-Round LOSS Attacks}
The honest-but-curious server follows FedAvg while inspecting returned client
weights and the resulting global model. For each labelled candidate $(x,y)$,
the membership score is $-\ell(f(x),y)$: larger scores indicate membership.
Only the first round is attacked. Global members are all 4,500 training records;
global nonmembers are the 4,500 held-out records. A local model's members are
only that client's training partition. \textbf{Local nonmembers are all records
outside that partition}, including other clients' training records and the
entire held-out set. Local member/nonmember counts are 4,500/4,500, 450/8,550,
225/8,775, and 90/8,910 for $K=1,10,20,50$ respectively.

We report ROC AUC and the greatest empirical TPR attainable with FPR at most
1\%, 5\%, or 10\%; no interpolated low-FPR point is presented as observed.
\textbf{Oracle-threshold balanced accuracy} is
$\max_t[\mathrm{TPR}(t)+1-\mathrm{FPR}(t)]/2$, using membership labels to choose
the best threshold retrospectively. It is descriptive, not the performance of
an attacker with an independently calibrated threshold. Every client's metric,
threshold, counts, score array, and ROC curve is preserved.
Client metrics are averaged within a run before aggregating three seed means;
individual clients are not treated as independent experimental replicates.

\begin{table}[ht]\centering\scriptsize
\caption{Baseline first-round attacks. Local rows are client means; entries are seed mean $\pm$ sample standard deviation.
BA denotes oracle-threshold balanced accuracy.}
\begin{tabular}{rlccccc}\toprule
$K$ & Model & Oracle BA & AUC & TPR@1\% & TPR@5\% & TPR@10\% \\
\midrule
@@ATTACKS@@
\bottomrule\end{tabular}\end{table}
\IfFileExists{figures/roc.png}{\begin{figure}[ht]\centering
\includegraphics[width=.91\linewidth]{roc.png}
\caption{Baseline empirical-step ROC averages. Global curves average seeds;
local curves average clients within seeds and then seeds.}\end{figure}}{}

@@LEAKAGE@@ Averaging can dilute individual records' influence, while small
client datasets and greater local work can increase record-specific fitting.
These are hypotheses to interpret the measured patterns, not guarantees.
Global and local attacks have different nonmember mixtures and class
compositions, so AUC differences cannot isolate aggregation as the sole cause.
First-round LOSS evaluates a narrow threat: near-chance results would not rule
out stronger attacks, later-round leakage, or leakage about real generator seeds.
@@PARAMETER_OBSERVATIONS@@

The vulnerable step is sending individual client weights to the curious server.
Transport encryption alone does not hide those weights from the receiving
server. Secure aggregation can reveal only the combined update, subject to an
appropriate aggregation cohort and dropout/collusion assumptions
\cite{secureagg}; a one-client cohort provides no such protection.
The global model can still leak membership. Clipping, calibrated noise, and a
privacy accountant are needed for a formal differential-privacy claim; noise
must protect updates before the curious server can inspect them, or be applied
inside a trusted aggregation boundary. Neither defense is evaluated here.

\section{Task C: Practicality and Future of FL}
\paragraph{Why adoption is difficult.}
FL's obstacles include heterogeneous and non-IID data, unreliable clients,
communication and device costs, deployment and debugging complexity, and
governance across organizations. Privacy requires additional mechanisms rather
than following automatically from decentralization \cite{survey}.
For generative models, large updates and heterogeneous client resources further
increase cost. The assignment's near-zero-deployment premise needs qualification:
Gboard has documented production FL with differential privacy, and Google
describes additional production federated-learning and analytics systems
\cite{gboard,parfait}. These examples do not establish broad adoption across industries.

\paragraph{What could increase adoption.}
Adoption becomes attractive when useful data cannot practically be pooled,
local data materially improve a model, and partners can share the infrastructure
and operational cost. Standardized schemas, reliable orchestration, auditable
privacy guarantees, and demonstrated utility gains can improve the business
case. These are reasoned adoption conditions, not measured forecasts.

\paragraph{Our healthcare startup.}
Cross-hospital FL could enable cooperation while retaining records locally,
but hospitals differ in coding, populations, and infrastructure. Our IID
synthetic simulation does not test those differences or establish clinical
validity. We would pursue a cross-silo pilot only with committed hospital
partners, agreed feature definitions, secure aggregation, explicit privacy
requirements, and evidence of value relative to simpler training arrangements.
Overfitting and the validation-selected models' low sensitivity require clinical
validation of training duration, calibration, and recall. AUROC or majority accuracy
alone cannot justify missed readmissions. Synthetic outcomes do not establish
clinical benefit.
The current insecure FedAvg implementation is an educational baseline, not a
deployment-ready privacy system.

\paragraph{Alternatives.}
If permitted and practical, a controlled centralized environment with appropriate
access restrictions and privacy-preserving training is simpler to operate.
Local-only models avoid sharing updates but lose cross-hospital learning.
Trusted execution environments offer another processing boundary but introduce
hardware and attestation assumptions. Synthetic data support development and
testing but do not themselves establish privacy or preserve clinical utility.
No alternative dominates: the choice depends on data access, trust, utility,
and operating cost \cite{survey,parfait}.

\textbf{Reproducibility and limits.}
Frozen data/configuration hashes, code, partitions, complete metrics, and tests
accompany this report; raw scores and checkpoints remain under \texttt{/work}.
Three seeds and synthetic IID data limit generalization. Attack assessment covers
only first-round LOSS.

\clearpage
\begin{thebibliography}{9}
\bibitem{uci} J. Clore, K. Cios, J. DeShazo, and B. Strack.
Diabetes 130-US Hospitals for Years 1999--2008. UCI, 2014.
\url{https://doi.org/10.24432/C5230J}.
\bibitem{tabddpm} A. Kotelnikov et al. TabDDPM: Modelling Tabular Data with Diffusion Models.
ICML, 2023. \url{https://arxiv.org/abs/2209.15421}.
\bibitem{fedavg} H. B. McMahan et al. Communication-Efficient Learning of Deep Networks
from Decentralized Data. AISTATS, 2017. \url{https://arxiv.org/abs/1602.05629}.
\bibitem{secureagg} K. Bonawitz et al. Practical Secure Aggregation for Privacy-Preserving
Machine Learning. CCS, 2017. \url{https://eprint.iacr.org/2017/281}.
\bibitem{survey} P. Kairouz et al. Advances and Open Problems in Federated Learning.
Foundations and Trends in Machine Learning, 2021. \url{https://arxiv.org/abs/1912.04977}.
\bibitem{gboard} Google Research. Improving Gboard language models via private federated
analytics. April 19, 2024.
\url{https://research.google/blog/improving-gboard-language-models-via-private-federated-analytics/}.
\bibitem{parfait} Google Research. Parfait: Enabling private AI with research tools.
January 22, 2025. \url{https://research.google/blog/parfait-enabling-private-ai-with-research-tools/}.
\bibitem{modelselection} Scikit-learn developers. Cross-validation: evaluating estimator performance;
MLPClassifier (validation-based early stopping). User Guide and API reference, accessed October 2026.
\url{https://scikit-learn.org/stable/modules/cross_validation.html}.
\url{https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html}.
\end{thebibliography}
\end{document}
"""
    substitutions = {"COVERAGE": coverage, "CONVERGENCE": convergence_text, "STATUS": status_note,
                     "PREVALENCE": prevalence_text, "PERFORMANCE": performance_table,
                     "EFFECTS": effects_table, "ATTACKS": attack_table, "LEAKAGE": leakage_text,
                     "TRAINING_OBSERVATIONS": training_observations,
                     "CONVERGENCE_OBSERVATIONS": convergence_observations,
                     "PILOT_OBSERVATIONS": pilot_observations,
                     "GENERALIZATION_FOLLOWUP": generalization_observations,
                     "EFFICIENCY_OBSERVATIONS": efficiency_observations,
                     "PARAMETER_OBSERVATIONS": parameter_observations}
    for key, value in substitutions.items():
        source = source.replace(f"@@{key}@@", value)
    (destination / "assignment_2_report.tex").write_text(source)
    slide = r"""\documentclass[aspectratio=169]{beamer}
\usepackage{graphicx}
\setbeamertemplate{navigation symbols}{}
\begin{document}
\begin{frame}{FedAvg and first-round membership inference}
\scriptsize \textbf{Team 08} --- Assignment lead: Chetan Kodand Reddy Madadi\\
Team members: Bharat Govil, Naveen Jarpla, Vishnu Fnu
\begin{columns}[T]
\begin{column}{.53\textwidth}
\begin{itemize}
\item Healthcare readmission: 9,000 regenerated synthetic encounters; 50/50 split.
\item 1/10/20/50 clients; batch/epoch/LR ablations; 28 configurations, three seeds.
\item @@UTILITY@@
@@GENERALIZATION_HEADLINE@@
\item @@HEADLINE@@
\item @@COVERAGE@@
\end{itemize}
\end{column}
\begin{column}{.45\textwidth}
\IfFileExists{figures/roc.png}{\includegraphics[width=\linewidth]{figures/roc.png}}{\textbf{Results pending}}
\scriptsize @@CONVERGENCE@@
\end{column}
\end{columns}
\medskip\footnotesize
Local nonmembers: every record outside that client's training partition.
Oracle-threshold BA uses membership labels retrospectively.
Secure aggregation hides individual updates; global-model privacy needs additional protection.
\end{frame}
\end{document}
"""
    for key, value in {"UTILITY": headline_utility, "HEADLINE": headline,
                       "COVERAGE": coverage, "CONVERGENCE": convergence_text,
                       "GENERALIZATION_HEADLINE": generalization_headline}.items():
        slide = slide.replace(f"@@{key}@@", value)
    (destination / "assignment_2_slide.tex").write_text(slide)


def render(destination, *original_args):
    """Keep the original analyzer callable; the default CLI renders the final campaign."""
    if original_args:
        return render_original(destination, *original_args)
    return render_final(Path(destination))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination',type=Path,default=HERE)
    args=p.parse_args();render(args.destination)
