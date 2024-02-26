import argparse
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PARENT_DIR)
sys.path.append(PARENT_DIR + "/compute_platform")
sys.path.append(PARENT_DIR + "/compute_platform/computing")
sys.path.append(PARENT_DIR + "/experiments")

from computing.proxy.model_proxy import ModelProxy
from container import (
    EarningCallFilterEGraph,
    EarningCallPrepocessor,
    MovementPredictorHeFF,
)
from ECHO_GL import HeFF


def exp_data_split(data_name="covid_sp500", exp_name="bt", init_window_size=6):
    data_split = {
        "maec_bt": {
            "start_index": init_window_size,
            "train": 590 - init_window_size + 1,
            # "train": 143 - init_window_size + 1,
            "test": 879 - 590,
        },
        "qin_bt": {
            "start_index": init_window_size,
            "train": 192 - init_window_size + 1,
            "test": 241 - 192,
        },
    }
    return data_split[data_name + "_" + exp_name]


def get_data(
    data_name,
    path_type,
    task="multitask",
    source_mode="multimodal",
    exp_name="bt",
    context_window_size=None,
    window_size=None,
    context_menu=None,
    predict_window_size=None,
    external_data_name=None,
):
    assert task in ["multitask"]
    assert source_mode in ["multimodal", "text_only", "audio_only"]
    if type(predict_window_size) != list:
        assert predict_window_size in [1, 3, 7, 15, 30]
    else:
        for w in predict_window_size:
            assert w in [1, 3, 7, 15, 30]

    if window_size is None:
        window_size = 30
    if context_window_size is None:
        context_window_size = 30

    data_split = exp_data_split(data_name, exp_name, init_window_size=window_size + 1)
    if context_menu is None:
        context_menu = [
            "series_ohlcv",
            "series_earning_call_audio",
            "series_earning_call_text",
        ]

    FILE_DIR = get_path_config(path_type, "dataset", data_name)
    external_data_path = (
        FILE_DIR + "/" + external_data_name + ".pt"
        if external_data_name is not None
        else None
    )

    data, stock_num = config_data(
        data_name=data_name,
        path_type=path_type,
        context_menu=context_menu,
        window_size=window_size,
        context_window_size=context_window_size,
        external_data_path=external_data_path,
    )
    metric, main_dir = config_metric(
        path_type=path_type, task=task, predict_window_size=predict_window_size
    )

    class_num = 2 if task == "movement" else 1
    if source_mode == "multimodal":
        use_text, use_audio = True, True
    elif source_mode == "text_only":
        use_text, use_audio = True, False
    elif source_mode == "audio_only":
        use_text, use_audio = False, True
    else:
        raise IOError(
            "Not support {} source_mode, please check the source_mode that you input!".format(
                task
            )
        )

    feature_dims = {
        "historical_price": 4,
        "earning_call_text": 768,
        "earning_call_audio": 29,
    }
    return (
        data,
        data_split,
        stock_num,
        metric,
        main_dir,
        feature_dims,
        window_size,
        class_num,
        use_text,
        use_audio,
    )


def config_metric(path_type="remote33", task="multitask", predict_window_size=None):
    if task == "multitask":
        metrics_class = {"HeFFEval": HeFFEval}
        metrics_name = {"HeFFEval": ["f1", "mcc"]}
        checkpoint_monitor = {
            "Decision": {"monitor_metric_name": "f1", "checkpoint_rule": "max"}
        }
        metrics_disable_mode = {
            "HeFFEval": {"PreTraining", "Training", "Updating", "Validation"}
        }
    else:
        raise IOError("Not support {} task, please check your task!".format(task))

    # Metric的参数 以及 实例生成

    metric_params = {
        "metrics_class": metrics_class,
        "metrics_name": metrics_name,
        "metrics_disable_mode": metrics_disable_mode,
        "checkpoint_monitor": checkpoint_monitor,
    }
    if task == "multitask":
        metric_params["predict_periods"] = predict_window_size
        metric_params["task_list"] = ["movement"]

    metric = (
        MetricProxy(**metric_params)
        if task != "multitask"
        else MultitaskMetricProxy(**metric_params)
    )
    main_dir = get_path_config(path_type, "metric")
    return metric, main_dir


def get_execute_set(
    data_split,
    save_metric,
    save_best_model,
    device,
    algorithm,
    main_dir,
    model,
    data,
    metric,
    task,
):
    execute_set_params = {
        "start_index": data_split["start_index"],
        "n_pretrain": 0,
        "n_train": data_split["train"],
        "n_validate": 0,
        "n_test": data_split["test"],
    }
    execute_set = ExecuteProxySet(**execute_set_params)

    execute_params = {
        "execute_proxy_class": "DLExecuteProxy",
        "result_dir": os.path.join(main_dir, "results/egraph_experiment/" + algorithm),
        "save_metric": save_metric,
        "save_best_model": save_best_model,
        "n_train_epoch": 1,
        "device": device,
        "task": "{}:{}".format(algorithm, task),
    }

    execute = DLExecuteProxy(model=model, data=data, metric=metric, **execute_params)
    execute_set.add_from_instance(execute)
    return execute, execute_set


def parse_args(to_add_args: list = None):
    """
    The default args include task, source_mode, duplicate, and data_name. If you want to add other args, please pass
    the to_add_args, which is a list of typle.

    Args:
        to_add_args(list of tuple): The arguments that your want to add.  The formate is [(option_string, dest,
        type, help, default), (option_string, dest, type, help, default)].
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        dest="task",
        type=str,
        help="select exp [dup, tune_beta, tune_k, without_prediction, without_relation]",
        default="movement",
    )
    parser.add_argument(
        "--source_mode",
        dest="source_mode",
        type=str,
        help="select source mode [multimodal, text_only, audio_only, price_only]",
        default="price_only",
    )

    parser.add_argument(
        "--duplicate",
        dest="duplicate",
        type=int,
        help="experiment duplicate",
        default=10,
    )
    parser.add_argument(
        "--data_name",
        dest="data_name",
        type=str,
        help="select exp [maec, qin]",
        default="maec",
    )

    if to_add_args is not None:
        for option_string, dest, type, help, default in to_add_args:
            parser.add_argument(
                option_string, dest=dest, type=type, help=help, default=default
            )

    args = parser.parse_args()
    return args


def run(data_name, path_type, save_metric=False, device="cuda", args=None):
    print("data_name:", data_name)
    predict_window_size = [1, 3, 7, 15, 30]
    # step 1. 获取数据

    if args.use_wiki_graph or args.use_price_graph or args.wo_hgt:
        context_menu = ["series_ohlcv", "series_earning_call_audio"]
    else:
        context_menu = [
            "series_ohlcv",
            "series_earning_call_audio",
            "series_earning_call_graph",
        ]

    external_data_name = "wiki_relation" if args.use_wiki_graph else None
    (
        data,
        data_split,
        stock_num,
        metric,
        main_dir,
        feature_dims,
        window_size,
        class_num,
        use_text,
        use_audio,
    ) = get_data(
        data_name=data_name,
        path_type=path_type,
        predict_window_size=predict_window_size,
        task="multitask",
        context_menu=context_menu,
        external_data_name=external_data_name,
        window_size=30,
        context_window_size=30,
    )

    # step 2. 创建model实例
    input_topic_dim = 556 if data_name == "qin" else 644
    input_wiki_relation_num = 16 if data_name == "qin" else 25
    net = HeFF(
        stock_num=stock_num,
        input_dim=768,
        input_topic_dim=input_topic_dim,
        lstm_dim=32,
        text_dim=16,
        entity_dim=4,
        topic_dim=4,
        lstm_layer=1,
        hgt_layer=2,
        divide_window=True,
        wo_hgt=args.wo_hgt,
        wo_sde=args.wo_sde,
        wo_entity=args.wo_entity,
        wo_topic=args.wo_topic,
        use_wiki_graph=args.use_wiki_graph,
        use_price_graph=args.use_price_graph,
        wiki_relation_num=input_wiki_relation_num,
    )

    filter = EarningCallFilterEGraph(batch_size=args.batch_size, predict_window_size=1)
    preprocessor = EarningCallPrepocessor(
        feature_dims=feature_dims, is_chunk_padding=False
    )
    modules_dict = {"net": net, "filter": filter, "preprocessor": preprocessor}
    # modules_dict = {"net": net, "preprocessor": preprocessor}
    #
    predictor = MovementPredictorHeFF(
        modules_dict=modules_dict,
        lr=args.lr,
        weight_decay=args.weight_decay,
        class_num=class_num,
        predict_window_size=predict_window_size,
        trading=args.trading,
    )

    model = ModelProxy(model=predictor)

    # step 3. 获取ExecuteSet
    algorithm = "heff"
    execute, execute_set = get_execute_set(
        data_split=data_split,
        save_metric=save_metric,
        save_best_model=False,
        device=device,
        algorithm=algorithm,
        main_dir=main_dir,
        model=model,
        data=data,
        metric=metric,
        task="{}_{}_{}".format(algorithm, "movement", predict_window_size),
    )

    # step 4. 配置运行pipeline
    n_train_epoch = 8

    # if data_name == "qin": n_train_epoch = 10
    for i in range(n_train_epoch):
        print("epoch: {}/{}".format(i, n_train_epoch))
        execute_set.execute(mode="Training", batch_size=args.batch_size)
        if (i + 1) % 1 == 0:
            execute_set.execute(mode="Decision", batch_size=1)
            if args.trading:
                execute_set.execute(
                    mode="Updating",
                    batch_size=1,
                    update_result={},
                    start_index=30,
                    end_index=31,
                )

    # execute_set.execute(mode='Decision', batch_size=1)
    execute.get_best_model_metric("Decision")


if __name__ == "__main__":
    path_type = "remote33"
    to_add_args = [
        ("--lr", "lr", float, "learning rate", 1e-3),
        ("--batch_size", "batch_size", int, "batch size", 32),
        ("--weight_decay", "weight_decay", float, "weight decay", 0.0),
        ("--wo_hgt", "wo_hgt", bool, "wo_hgt", False),
        ("--wo_sde", "wo_sde", bool, "wo_sde", False),
        ("--wo_entity", "wo_entity", bool, "", False),
        ("--wo_topic", "wo_topic", bool, "", True),
        ("--use_wiki_graph", "use_wiki_graph", bool, "", False),
        ("--use_price_graph", "use_price_graph", bool, "", True),
        ("--trading", "trading", bool, "", True),
    ]

    args = parse_args(to_add_args)
    args.data_name = "qin"
    print(args)
    # ==================================== 打包带走 ================================
    # run_task(args.data_name, path_type, duplicate=args.duplicate, save_metric=True, run_single=run_single,
    #          task=args.task, args=args)

    # ==================================== 单次重复实验 ================================
    # predict_window_size = 1  # 1, 3,7,15,30
    # run_single(data_name, path_type, "volatility", predict_window_size, duplicate=1, save_metric=True)

    # ==================================== 单次实验 ================================
    # predict_window_size = 3  # 1, 3,7,15,30
    run(args.data_name, path_type, save_metric=True, args=args)
