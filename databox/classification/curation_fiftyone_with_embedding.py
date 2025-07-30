# %%
import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.types as fot
import fiftyone.zoo as foz

# %%
fo.annotation_config.backends["cvat"]["segment_size"] = 300
# %%
dataset_dir = "/home/fkwong/datasets/82_truckcls/data/raw/truckcls-fiftyone"
dataset_type = fot.FiftyOneDataset
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
label_field = "ground_truth"  # classification
temp_anno_field = "temp_annotation"
# %%
dataset = fo.Dataset.from_dir(
    dataset_type=dataset_type,
    dataset_dir=dataset_dir,
)
# %%
model = foz.load_zoo_model("clip-vit-base32-torch")
dataset.compute_embeddings(
    model, embeddings_field="clip-vit-base32-torch-embeddings", progress=True
)
# %%
visualization_result = fob.compute_visualization(
    dataset,
    embeddings="clip-vit-base32-torch-embeddings",
    brain_key="clip_vit_base32_torch_visualization",
    progress=True,
)
# %%
similarity_result = fob.compute_similarity(
    dataset,
    embeddings="clip-vit-base32-torch-embeddings",
    brain_key="clip_vit_base32_torch_similarity",
    progress=True,
)
# %%
sess = fo.Session(dataset, auto=False)
# %%
anno_view = dataset.match_tags("relabel")
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
