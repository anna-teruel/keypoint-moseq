import jax  # type: ignore
import keypoint_moseq as kpms
import numpy as np

project_dir = '/home/vteruel/anna/kpms/kpms_projects/kpms-3cham/'
config = lambda: kpms.load_config(project_dir)
keypoint_data_path = '/home/vteruel/anna/kpms/kpms_projects/kpms-3cham/dlc'
coordinates, confidences, bodyparts = kpms.load_keypoints(keypoint_data_path, 'deeplabcut')
data, metadata = kpms.format_data(coordinates, confidences, **config())

# PCA
pca = kpms.fit_pca(**data, **config())
kpms.save_pca(pca, project_dir)

# Model comparison
num_model_fits = 20
prefix = 'models'

ar_only_kappa = 1e7
num_ar_iters = 50

full_model_kappa = 1e7
num_full_iters = 500

for restart in range(num_model_fits):
    print(f"Fitting model {restart}")
    model_name = f'{prefix}-{restart}'

    model = kpms.init_model(
        data, pca=pca, **config(), seed=jax.random.PRNGKey(restart), location_aware=True
    )

    # stage 1: fit the model with AR only
    model = kpms.update_hypparams(model, kappa=ar_only_kappa)
    model = kpms.fit_model(
        model,
        data,
        metadata,
        project_dir,
        model_name,
        ar_only=True,
        num_iters=num_ar_iters, 
       # location_aware=True
    )[0]

    # stage 2: fit the full model
    from jax_moseq.utils import set_mixed_map_iters
    set_mixed_map_iters(4)
    
    model = kpms.update_hypparams(model, kappa=full_model_kappa)
    kpms.fit_model(
        model,
        data,
        metadata,
        project_dir,
        model_name,
        ar_only=False,
        start_iter=num_ar_iters,
        num_iters=num_full_iters, 
        #location_aware=True
    );

    kpms.reindex_syllables_in_checkpoint(project_dir, model_name);
    model, data, metadata, current_iter = kpms.load_checkpoint(project_dir, model_name)
    results = kpms.extract_results(model, metadata, project_dir, model_name)

model_names = ['models-{}'.format(i) for i in range(20)]

eml_scores, eml_std_errs = kpms.expected_marginal_likelihoods(project_dir, model_names)
best_model = model_names[np.argmax(eml_scores)]
print(f"Best model: {best_model}")

kpms.plot_eml_scores(eml_scores, eml_std_errs, model_names)
