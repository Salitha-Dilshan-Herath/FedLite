from django.shortcuts import render
from django.http import JsonResponse
import uuid, threading
import pandas as pd

from .forms import ExperimentForm
from .fl_engine import run_experiment, RunConfig
from .plot_utils import plot_lines
from .job_store import create_job, set_job_phase, finish_job, fail_job, get_job


EXPERIMENTS = [
    {"key": "fedavg", "name": "FedAvg (E=1)", "phase": "exp_0", "cfg": lambda le: RunConfig(local_epochs=1), "strategy": "fedavg"},
    {"key": "fedavg_e2", "name": "FedAvg (E=2)", "phase": "exp_1", "cfg": lambda le: RunConfig(local_epochs=2), "strategy": "fedavg"},
    {"key": "fedprox", "name": "FedProx", "phase": "exp_2", "cfg": lambda le: RunConfig(local_epochs=1, proximal_mu=0.01), "strategy": "fedprox"},
    {"key": "fedadam", "name": "FedAdam", "phase": "exp_3", "cfg": lambda le: RunConfig(local_epochs=1), "strategy": "fedadam"},
]


def index(request):
    form = ExperimentForm()
    return render(request, "dashboard/index.html", {"form": form})


def start_run(request):

    dataset_name = request.POST.get("dataset", "cifar10")
    rounds = int(request.POST.get("rounds", 30))
    num_clients = int(request.POST.get("num_clients", 20))
    fraction_fit = float(request.POST.get("fraction_fit", 0.2))
    alpha = float(request.POST.get("alpha", 0.5))

    topk_percent = float(request.POST.get("topk_percent", 5.0))
    quant_bits = int(request.POST.get("quant_bits", 8))
    local_epochs = int(request.POST.get("local_epochs", 2))
    ef = request.POST.get("error_feedback", "on") == "on"

    job_id = str(uuid.uuid4())
    strat_name = f"{dataset_name.upper()}-TopK({topk_percent}%)"

    experiment_names = [e["name"] for e in EXPERIMENTS] + [strat_name]
    create_job(job_id, total_rounds=rounds, experiment_names=experiment_names)

    def worker():
        try:
            dfs = []
            results_list = []

            for i, exp in enumerate(EXPERIMENTS):
                set_job_phase(job_id, exp["phase"])
                cfg = exp["cfg"](local_epochs)
                df = run_experiment(
                    name=exp["name"],
                    num_clients=num_clients,
                    rounds=rounds,
                    fraction_fit=fraction_fit,
                    alpha=alpha,
                    cfg=cfg,
                    dataset_name=dataset_name,
                    job_id=job_id,
                    experiment_phase=exp["phase"],
                    strategy_type=exp["strategy"],
                )
                dfs.append(df)
                last = df.sort_values("round").tail(1).iloc[0]
                results_list.append({
                    "name": exp["name"],
                    "final_acc": float(last["accuracy"]),
                    "upload_mb": float(last["upload_mb_cum"]),
                    "avg_upload_mb": float(last["upload_mb_avg"]),
                })

            set_job_phase(job_id, "exp_4")
            df_strat = run_experiment(
                name=strat_name,
                num_clients=num_clients,
                rounds=rounds,
                fraction_fit=fraction_fit,
                alpha=alpha,
                cfg=RunConfig(
                    local_epochs=local_epochs,
                    quant_bits=quant_bits,
                    topk_frac=topk_percent / 100.0,
                    error_feedback=ef,
                ),
                dataset_name=dataset_name,
                job_id=job_id,
                experiment_phase="exp_4",
                strategy_type="fedavg",
            )
            dfs.append(df_strat)
            strat_last = df_strat.sort_values("round").tail(1).iloc[0]
            results_list.append({
                "name": strat_name,
                "final_acc": float(strat_last["accuracy"]),
                "upload_mb": float(strat_last["upload_mb_cum"]),
                "avg_upload_mb": float(strat_last["upload_mb_avg"]),
            })

            df_cmp = pd.concat(dfs)

            acc_img = plot_lines(df_cmp, "round", "accuracy", "Accuracy", "Round", "Accuracy")
            up_img = plot_lines(df_cmp, "round", "upload_mb_avg", "Upload per Round", "Round", "MB")
            cum_img = plot_lines(df_cmp, "round", "upload_mb_cum", "Cumulative Upload", "Round", "MB")

            result = {
                "experiments": results_list,
                "acc_img": acc_img,
                "up_img": up_img,
                "cum_img": cum_img,
            }

            finish_job(job_id, result)

        except Exception as e:
            fail_job(job_id, str(e))

    threading.Thread(target=worker, daemon=True).start()
    return JsonResponse({"job_id": job_id})


def progress(request, job_id):
    job = get_job(job_id)
    if not job:
        return JsonResponse({"error": "job not found"}, status=404)

    return JsonResponse({
        "status": job.status,
        "current_round": job.current_round,
        "total_rounds": job.total_rounds,
        "message": job.message,
        "result": job.result,
        "error": job.error,
        "phase": job.phase,
        "phase_index": job.phase_index,
        "experiments": job.experiments,
    })
