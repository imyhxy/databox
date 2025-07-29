# %%
import os.path as osp
import pickle

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.types as fot
from fiftyone import ViewField as F

# %%
fo.annotation_config.backends["cvat"]["segment_size"] = 300
# %%
dataset_dir = "/home/fkwong/datasets/82_truckcls/data/raw/truckcls-fiftyone"
dataset_type = fot.FiftyOneImageClassificationDataset
prediction_file = "/home/fkwong/workspace/srcs/public/grp_openmm/mmpretrain/work_dirs/test_outputs/efficientnet-b1_8xb32_truckcls/pred.pkl"
categories = [
    "danger_vehicle",
    "dumper",
    "dumper6",
    "others",
    "pickup",
    "truck_box",
    "unknown",
]
anno_key = "cvat_annotation"
prediction_field = "efficientnet_b1"
label_field = "ground_truth"  # classification
temp_anno_field = "temp_annotation"
# %%
dataset = fo.Dataset.from_dir(
    dataset_type=dataset_type,
    dataset_dir=dataset_dir,
)
# %%
predictions = pickle.load(open(prediction_file, "rb"))
prediction_map = {}
for prediction in predictions:
    filepath = osp.abspath(prediction["img_path"])
    prediction_map[filepath] = prediction
# %%
with dataset.save_context() as ctx:
    for sample in dataset:
        filepath = osp.abspath(sample.filepath)
        if filepath in prediction_map:
            prediction = prediction_map[filepath]
            idx = prediction["pred_label"].item()
            label = categories[idx]
            sample[prediction_field] = fo.Classification(
                label=label,
                confidence=prediction["pred_score"][idx].item(),
                logits=prediction["pred_score"].numpy().tolist(),
            )
            ctx.save(sample)
# %%
prediction_view = dataset.match(~F(prediction_field).is_null())
# %%
mistakenness_field = f"{prediction_field}_mistakenness"
fob.compute_mistakenness(
    prediction_view,
    prediction_field,
    "ground_truth",
    mistakenness_field=mistakenness_field,
)
anno_view = prediction_view.match(F(mistakenness_field) > 0.90)
# %%
sess = fo.Session(anno_view, auto=False)
# %%
anno_view = dataset.select(sess.selected)
# %%
if len(anno_view) > 0:
    if anno_key in dataset.list_saved_views():
        print(f"{anno_key} view existed!")
    else:
        dataset.save_view(anno_key, anno_view)
else:
    print("No sample to be annotated.")
# %%
if len(anno_view) > 0:
    sess = fo.Session(anno_view, auto=False)
else:
    print("No sample to be reviewed.")
# %%
if len(anno_view) > 0:
    anno_results = anno_view.annotate(
        anno_key=anno_key,
        label_field=label_field,
        label_type="classification",
        classes=categories,
        launch_editor=False,
    )
else:
    print("No sample to be annotated.")
# %%
anno_view = dataset.load_annotation_view(anno_key)
anno_view.load_annotations(
    anno_key,
    dest_field=temp_anno_field,
    cleanup=False,
)
# %%
sess.view = anno_view
# %%
anno_results = anno_view.load_annotation_results(anno_key)
for k, v in anno_results.get_status()[label_field].items():
    if v["status"] != "completed":
        print(f"Task-{k} is not completed yet, current status: {v['status']}")
        break
else:
    anno_view.load_annotations(anno_key, dest_field=label_field, cleanup=False)
# %%
anno_results = dataset.load_annotation_results(
    anno_key,
    cache=False,
)
# %%
dataset.delete_sample_fields(
    [
        temp_anno_field,
        mistakenness_field,
        prediction_field,
    ],
    error_level=1,
)
# %%
dataset.export(
    export_dir=dataset_dir,
    dataset_type=dataset_type,
    export_media=True,
)
# %%
anno_results.cleanup()
dataset.delete_annotation_run(anno_key)
dataset.delete_saved_view(anno_key)
dataset.list_saved_views()
dataset.list_annotation_runs()

# %%
