# %%
import warnings

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.types as fot
import fiftyone.zoo as foz
import patchbrain  # noqa: F401

warnings.filterwarnings(action="ignore", category=UserWarning)
# %%
dataset_dir = "/home/fkwong/datasets/82_truckcls/data/raw/truckcls-fiftyone"
output_dri = "/home/fkwong/datasets/82_truckcls/data/raw/truckcls-fiftyone-new"
dataset_type = fot.FiftyOneDataset
# %%
dataset = fo.Dataset.from_dir(dataset_type=dataset_type, dataset_dir=dataset_dir)
sess = fo.Session(dataset, auto=False)


# %%
def get_similarity(cfg, dataset):
    if dataset.has_brain_run(cfg["sim_key"]):
        sim_res = dataset.load_brain_results(cfg["sim_key"])
    else:
        if not dataset.has_sample_field(cfg["embed_key"]):
            model = foz.load_zoo_model(cfg["model_name"])
            dataset.compute_embeddings(
                model=model,
                embeddings_field=cfg["embed_key"],
                batch_size=cfg["batch_size"],
                num_workers=cfg["num_workers"],
            )
        sim_res = fob.compute_similarity(
            dataset, brain_key=cfg["sim_key"], embeddings=cfg["embed_key"]
        )
    sim_res.find_duplicates(thresh=cfg["dup_thresh"])
    dup_view = sim_res.duplicates_view(
        type_field=cfg["type_field"],
        id_field=cfg["id_field"],
        dist_field=cfg["dist_field"],
        reverse=True,
    )
    return sim_res, dup_view


# %%
def get_model_config(name):
    if name == "clip-vit-base32-torch":
        # Settings for clip-vit-base32-torch
        return dict(  # noqa: C408
            model_name=name,
            batch_size=64,
            num_workers=8,
            embed_key="clip_vit_embeddings",
            uni_key="clip_vit_uniqueness",
            sim_key="clip_vit_similarity",
            viz_key="clip_vit_visualization",
            type_field="clip_vit_dup_type",
            id_field="clip_vit_dup_id",
            dist_field="clip_vit_dup_dist",
            viz_seed=1234,
            dup_thresh=0.03,  # modify according to experience, 0.05 - 0.08
            duplicate_tag="clip_vit_duplicate",
            unique_tag="clip_vit_unique",
        )
    else:
        raise AssertionError


# %%
cfg = get_model_config("clip-vit-base32-torch")
sim_res, dup_view = get_similarity(cfg, dataset)
sess.view = dup_view
print(len(sim_res.duplicate_ids))
sess.open_tab()
# %%
dataset.delete_samples(dataset.select(sim_res.duplicate_ids))
dataset.export(export_dir=output_dri, dataset_type=dataset_type, export_media=True)

# %%
