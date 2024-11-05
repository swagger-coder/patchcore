import contextlib
import logging
import os
import sys

import click
import numpy as np
import torch

import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score

import patchcore.backbones
import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils

LOGGER = logging.getLogger(__name__)

# _DATASETS = {"mvtec": ["patchcore.datasets.mvtec", "MVTecDataset"]}
_DATASETS = {"huawei": ["patchcore.datasets.huawei", "HuiweiDataset"]}


@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str, default="group")
@click.option("--log_project", type=str, default="project")
@click.option("--save_segmentation_images", is_flag=True)
@click.option("--save_patchcore_model", is_flag=True)
def main(**kwargs):
    pass


@main.result_callback()
def run(
    methods,
    results_path,
    gpu,
    seed,
    log_group,
    log_project,
    save_segmentation_images,
    save_patchcore_model,
):
    methods = {key: item for (key, item) in methods}

    run_save_path = patchcore.utils.create_storage_folder(
        results_path, log_project, log_group, mode="iterate"
    )

    list_of_dataloaders = methods["get_dataloaders"](seed)

    device = patchcore.utils.set_torch_device(gpu)
    # Device context here is specifically set and used later
    # because there was GPU memory-bleeding which I could only fix with
    # context managers.
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    result_collect = []

    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["training"].name,
                dataloader_count + 1,
                len(list_of_dataloaders),
            )
        )

        patchcore.utils.fix_seeds(seed, device)

        dataset_name = dataloaders["training"].name

        with device_context:
            torch.cuda.empty_cache()
            imagesize = dataloaders["training"].dataset.imagesize
            sampler = methods["get_sampler"](
                device,
            )
            PatchCore_list = methods["get_patchcore"](imagesize, sampler, device)
            if len(PatchCore_list) > 1:
                LOGGER.info(
                    "Utilizing PatchCore Ensemble (N={}).".format(len(PatchCore_list))
                )
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                if PatchCore.backbone.seed is not None:
                    patchcore.utils.fix_seeds(PatchCore.backbone.seed, device)
                LOGGER.info(
                    "Training models ({}/{})".format(i + 1, len(PatchCore_list))
                )
                torch.cuda.empty_cache()
                PatchCore.fit(dataloaders["training"])

            torch.cuda.empty_cache()
            aggregator = {"scores": [], "segmentations": []}
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                LOGGER.info(
                    "Embedding test data with models ({}/{})".format(
                        i + 1, len(PatchCore_list)
                    )
                )
                scores, segmentations, labels_gt, masks_gt = PatchCore.predict(
                    dataloaders["testing"]
                )
                aggregator["scores"].append(scores)
                aggregator["segmentations"].append(segmentations)

            scores = np.array(aggregator["scores"])
            min_scores = scores.min(axis=-1).reshape(-1, 1)
            max_scores = scores.max(axis=-1).reshape(-1, 1)
            scores = (scores - min_scores) / (max_scores - min_scores)
            scores = np.mean(scores, axis=0)

            segmentations = np.array(aggregator["segmentations"])
            min_scores = (
                segmentations.reshape(len(segmentations), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                segmentations.reshape(len(segmentations), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            segmentations = (segmentations - min_scores) / (max_scores - min_scores)
            segmentations = np.mean(segmentations, axis=0)

            anomaly_labels = [
                x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate
            ]

            # (Optional) Plot example images.
            if save_segmentation_images:
                image_paths = [
                    x[2] for x in dataloaders["testing"].dataset.data_to_iterate
                ]
                mask_paths = [
                    x[3] for x in dataloaders["testing"].dataset.data_to_iterate
                ]

                def image_transform(image):
                    in_std = np.array(
                        dataloaders["testing"].dataset.transform_std
                    ).reshape(-1, 1, 1)
                    in_mean = np.array(
                        dataloaders["testing"].dataset.transform_mean
                    ).reshape(-1, 1, 1)
                    image = dataloaders["testing"].dataset.transform_img(image)
                    return np.clip(
                        (image.numpy() * in_std + in_mean) * 255, 0, 255
                    ).astype(np.uint8)

                def mask_transform(mask):
                    return dataloaders["testing"].dataset.transform_mask(mask).numpy()

                image_save_path = os.path.join(
                    run_save_path, "segmentation_images", dataset_name
                )
                os.makedirs(image_save_path, exist_ok=True)
                patchcore.utils.plot_segmentation_images(
                    image_save_path,
                    image_paths,
                    segmentations,
                    scores,
                    mask_paths,
                    image_transform=image_transform,
                    mask_transform=mask_transform,
                )

            LOGGER.info("Computing evaluation metrics.")
            auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
                scores, anomaly_labels
            )["auroc"]

            

            def cal_metrics(gt_label, score_label, file_path, img_names_list: list = None, threshold: float = None):
                """
                计算并保存二分类的评估指标.

                参数:
                gt_label (array-like): 真实的二分类标签 (0 或 1)，也可以是 True/False. 其中请确保正样本为1
                score_label (array-like): 预测的分数或概率.
                file_path (str): 保存结果的Excel文件路径.
                img_names_list (list): 图片名列表，用于保存json文件.
                threshold (float): 阈值，传入后额外计算指标.

                输出:
                Excel文件，包含以下列: 精确度, 召回率, FPR (过杀率), FNR (漏杀率), AUROC (ROC曲线下面积), 最佳阈值.
                如果传入图片名列表，则额外保存一个json文件，包含每张图片的预测结果.
                如果传入阈值，则额外保存一个文件，包含精确度, 召回率, FPR, FNR, AUROC, 最佳阈值
                """
                gt_label = np.array(gt_label, dtype=int)
                precision, recall, thresholds = precision_recall_curve(gt_label, score_label)
                a = 2 * precision * recall
                b = precision + recall
                f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
                max_f1_index = np.argmax(f1)
                det_threshold = thresholds[max_f1_index]
                
                predictions = (score_label >= det_threshold).astype(int)

                tp = np.sum((predictions == 1) & (gt_label == 1))
                fp = np.sum((predictions == 1) & (gt_label == 0))
                fn = np.sum((predictions == 0) & (gt_label == 1))
                tn = np.sum((predictions == 0) & (gt_label == 0))
                
                # 计算FPR和FNR，避免除以零
                fpr_value = fp / (fp + tn) if (fp + tn) > 0 else 0
                fnr_value = fn / (tp + fn) if (tp + fn) > 0 else 0

                # 计算AUROC
                auroc_value = roc_auc_score(gt_label, score_label)

                # 打印最佳阈值及其对应的指标
                print(f"根据F1分数选择的最佳阈值: {det_threshold:.2f}")
                print(f"最佳阈值对应的精确度: {precision[max_f1_index]:.2f}")
                print(f"最佳阈值对应的召回率: {recall[max_f1_index]:.2f}")
                print(f"最佳阈值对应的F1分数: {f1[max_f1_index]:.2f}")
                print(f"FPR: {fpr_value:.2f}")
                print(f"FNR: {fnr_value:.2f}")
                print(f"AUROC: {auroc_value:.2f}")

                # 将结果保存到Excel
                results = {
                    '精确度': [precision[max_f1_index]],
                    '召回率': [recall[max_f1_index]],
                    'FPR': [fpr_value],
                    'FNR': [fnr_value],
                    'AUROC': [auroc_value],
                    '阈值': [det_threshold]
                }
                df = pd.DataFrame(results)
                df.to_excel(file_path, index=False)


                # 根据阈值计算指标
                if threshold != None:
                    predictions = (score_label >= threshold).astype(int)
                    tp = np.sum((predictions == 1) & (gt_label == 1))
                    fp = np.sum((predictions == 1) & (gt_label == 0))
                    fn = np.sum((predictions == 0) & (gt_label == 1))
                    tn = np.sum((predictions == 0) & (gt_label == 0))
                    
                    # 计算FPR和FNR，避免除以零
                    fpr_value = fp / (fp + tn) if (fp + tn) > 0 else 0
                    fnr_value = fn / (tp + fn) if (tp + fn) > 0 else 0

                    # 计算AUROC
                    auroc_value = roc_auc_score(gt_label, score_label)

                    # 根据threshold计算精确度、召回率、F1分数
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                

                    # 打印最佳阈值及其对应的指标
                    print(f"传入固定阈值: {threshold:.2f}")
                    print(f"固定阈值对应的精确度: {precision:.2f}")
                    print(f"固定阈值对应的召回率: {recall:.2f}")
                    print(f"固定阈值对应的F1分数: {f1:.2f}")
                    print(f"FPR: {fpr_value:.2f}")
                    print(f"FNR: {fnr_value:.2f}")
                    print(f"AUROC: {auroc_value:.2f}")

                    # 将结果保存到Excel
                    results = {
                        '精确度': [precision[max_f1_index]],
                        '召回率': [recall[max_f1_index]],
                        'FPR': [fpr_value],
                        'FNR': [fnr_value],
                        'AUROC': [auroc_value],
                        '阈值': [threshold]
                    }
                    df = pd.DataFrame(results)
                    df.to_excel(file_path.replace(".xlsx", f"_threshold_{threshold:.2f}.xlsx"), index=False)


                # 保存中间结果
                result_all = []

                if img_names_list != None:
                    for i in range(len(gt_label)):
                        result = {}
                        result["image name"] = img_names_list[i]
                        result["pred"] = int(predictions[i])
                        result["gt"] = int(gt_label[i])
                        result["threshold"] = float(det_threshold)
                        result["score"] = float(score_label[i])
                        result_all.append(result)
                        
                    # 保存json
                    
                    with open(file_path.replace(".xlsx", ".json"), "w", encoding='utf-8') as f:
                        json.dump(result_all, f, ensure_ascii=False)

            
            if dataloaders["testing"].dataset.classnames_to_use[0] == 'leisi':
                cal_metrics(anomaly_labels, scores, f'{dataloaders["testing"].dataset.classnames_to_use[0]}_{dataloaders["testing"].dataset.split.value}.xlsx')
            else:
                cal_metrics(anomaly_labels, scores, f'{dataloaders["testing"].dataset.classnames_to_use[0]}_{dataloaders["testing"].dataset.split.value}.xlsx')

            # Compute PRO score & PW Auroc for all images
            pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                segmentations, masks_gt
            )
            full_pixel_auroc = pixel_scores["auroc"]

            # Compute PRO score & PW Auroc only images with anomalies
            sel_idxs = []
            for i in range(len(masks_gt)):
                if np.sum(masks_gt[i]) > 0:
                    sel_idxs.append(i)
            pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                [segmentations[i] for i in sel_idxs],
                [masks_gt[i] for i in sel_idxs],
            )
            anomaly_pixel_auroc = pixel_scores["auroc"]

            result_collect.append(
                {
                    "dataset_name": dataset_name,
                    "instance_auroc": auroc,
                    "full_pixel_auroc": full_pixel_auroc,
                    "anomaly_pixel_auroc": anomaly_pixel_auroc,
                }
            )

            for key, item in result_collect[-1].items():
                if key != "dataset_name":
                    LOGGER.info("{0}: {1:3.3f}".format(key, item))

            # (Optional) Store PatchCore model for later re-use.
            # SAVE all patchcores only if mean_threshold is passed?
            if save_patchcore_model:
                patchcore_save_path = os.path.join(
                    run_save_path, "models", dataset_name
                )
                os.makedirs(patchcore_save_path, exist_ok=True)
                for i, PatchCore in enumerate(PatchCore_list):
                    prepend = (
                        "Ensemble-{}-{}_".format(i + 1, len(PatchCore_list))
                        if len(PatchCore_list) > 1
                        else ""
                    )
                    PatchCore.save_to_path(patchcore_save_path, prepend)

        LOGGER.info("\n\n-----\n")

    # Store all results and mean scores to a csv-file.
    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results["dataset_name"] for results in result_collect]
    result_scores = [list(results.values())[1:] for results in result_collect]
    patchcore.utils.compute_and_store_final_results(
        run_save_path,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
    )


@main.command("patch_core")
# Pretraining-specific parameters.
@click.option("--backbone_names", "-b", type=str, multiple=True, default=[])
@click.option("--layers_to_extract_from", "-le", type=str, multiple=True, default=[])
# Parameters for Glue-code (to merge different parts of the pipeline.
@click.option("--pretrain_embed_dimension", type=int, default=1024)
@click.option("--target_embed_dimension", type=int, default=1024)
@click.option("--preprocessing", type=click.Choice(["mean", "conv"]), default="mean")
@click.option("--aggregation", type=click.Choice(["mean", "mlp"]), default="mean")
# Nearest-Neighbour Anomaly Scorer parameters.
@click.option("--anomaly_scorer_num_nn", type=int, default=5)
# Patch-parameters.
@click.option("--patchsize", type=int, default=3)
@click.option("--patchscore", type=str, default="max")
@click.option("--patchoverlap", type=float, default=0.0)
@click.option("--patchsize_aggregate", "-pa", type=int, multiple=True, default=[])
# NN on GPU.
@click.option("--faiss_on_gpu", is_flag=True)
@click.option("--faiss_num_workers", type=int, default=8)
def patch_core(
    backbone_names,
    layers_to_extract_from,
    pretrain_embed_dimension,
    target_embed_dimension,
    preprocessing,
    aggregation,
    patchsize,
    patchscore,
    patchoverlap,
    anomaly_scorer_num_nn,
    patchsize_aggregate,
    faiss_on_gpu,
    faiss_num_workers,
):
    backbone_names = list(backbone_names)
    if len(backbone_names) > 1:
        layers_to_extract_from_coll = [[] for _ in range(len(backbone_names))]
        for layer in layers_to_extract_from:
            idx = int(layer.split(".")[0])
            layer = ".".join(layer.split(".")[1:])
            layers_to_extract_from_coll[idx].append(layer)
    else:
        layers_to_extract_from_coll = [layers_to_extract_from]

    def get_patchcore(input_shape, sampler, device):
        loaded_patchcores = []
        for backbone_name, layers_to_extract_from in zip(
            backbone_names, layers_to_extract_from_coll
        ):
            backbone_seed = None
            if ".seed-" in backbone_name:
                backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(
                    backbone_name.split("-")[-1]
                )
            backbone = patchcore.backbones.load(backbone_name)
            backbone.name, backbone.seed = backbone_name, backbone_seed

            nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)

            patchcore_instance = patchcore.patchcore.PatchCore(device)
            patchcore_instance.load(
                backbone=backbone,
                layers_to_extract_from=layers_to_extract_from,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=target_embed_dimension,
                patchsize=patchsize,
                featuresampler=sampler,
                anomaly_scorer_num_nn=anomaly_scorer_num_nn,
                nn_method=nn_method,
            )
            loaded_patchcores.append(patchcore_instance)
        return loaded_patchcores

    return ("get_patchcore", get_patchcore)


@main.command("sampler")
@click.argument("name", type=str)
@click.option("--percentage", "-p", type=float, default=0.1, show_default=True)
def sampler(name, percentage):
    def get_sampler(device):
        if name == "identity":
            return patchcore.sampler.IdentitySampler()
        elif name == "greedy_coreset":
            return patchcore.sampler.GreedyCoresetSampler(percentage, device)
        elif name == "approx_greedy_coreset":
            return patchcore.sampler.ApproximateGreedyCoresetSampler(percentage, device)

    return ("get_sampler", get_sampler)


@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--train_val_split", type=float, default=1, show_default=True)
@click.option("--batch_size", default=8, type=int, show_default=True)
@click.option("--num_workers", default=8, type=int, show_default=True)
@click.option("--resize", default=256, type=int, show_default=True)
@click.option("--imagesize", default=224, type=int, show_default=True)
@click.option("--augment", is_flag=True)
def dataset(
    name,
    data_path,
    subdatasets,
    train_val_split,
    batch_size,
    resize,
    imagesize,
    num_workers,
    augment,
):
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders(seed):
        dataloaders = []
        for subdataset in subdatasets:
            train_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                train_val_split=train_val_split,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TRAIN,
                seed=seed,
                augment=augment,
            )

            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
            )

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            train_dataloader.name = name
            if subdataset is not None:
                train_dataloader.name += "_" + subdataset

            if train_val_split < 1:
                val_dataset = dataset_library.__dict__[dataset_info[1]](
                    data_path,
                    classname=subdataset,
                    resize=resize,
                    train_val_split=train_val_split,
                    imagesize=imagesize,
                    split=dataset_library.DatasetSplit.VAL,
                    seed=seed,
                )

                val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
            else:
                val_dataloader = None
            dataloader_dict = {
                "training": train_dataloader,
                "validation": val_dataloader,
                "testing": test_dataloader,
            }

            dataloaders.append(dataloader_dict)
        return dataloaders

    return ("get_dataloaders", get_dataloaders)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()
