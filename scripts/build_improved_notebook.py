"""Generate notebooks/thyroid_improved.ipynb (a thin, readable front-end to the package; no logic is duplicated)."""
import os
import sys

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MD = nbf.v4.new_markdown_cell
CODE = nbf.v4.new_code_cell

cells = [
    MD("# Radiation-induced hypothyroidism: leakage-safe nested-CV analysis\n"
       "Front-end notebook for the `thyroid_rihp` package. The original notebook "
       "(`thyroid-6292026.ipynb`) is preserved unchanged; see `reports/AUDIT_REPORT.md` for what was fixed.\n\n"
       "**All results below are internal (nested, repeated, stratified) cross-validation on 115 patients / 27 events. "
       "There is no external validation.**"),
    CODE("import os, sys, warnings\n"
         "os.environ.setdefault('OMP_NUM_THREADS', '1')\n"
         "ROOT = os.path.abspath('..')\n"
         "sys.path.insert(0, ROOT)\n"
         "warnings.filterwarnings('ignore')\n"
         "import pandas as pd\n"
         "from IPython.display import Image, display\n"
         "from thyroid_rihp.config import load_config, resolve\n"
         "from thyroid_rihp.data import load_dataset, feature_group_combinations, combo_name\n"
         "pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)\n"
         "cfg = load_config(os.path.join(ROOT, 'config', 'config.yaml'))\n"
         "cfg['paths']['data_dir'] = os.path.join(ROOT, 'data'); cfg['paths']['output_dir'] = os.path.join(ROOT, 'outputs')\n"
         "RUN = resolve(cfg, 'output_dir') / 'main'"),
    MD("## 1. Data validation"),
    CODE("ds = load_dataset(cfg)\n"
         "v = {k: x for k, x in ds.validation.items() if k != 'files'}\n"
         "pd.Series({k: v[k] for k in ['n_patients','n_positive','n_negative','prevalence','n_features_per_group',\n"
         "          'label_consistent_across_files','ids_identical_across_files','outcome_sorted_by_row_order',\n"
         "          'rows_sharing_identical_name_in_CB','name_mismatch_ids_vs_CB']})"),
    CODE("labels = cfg['data']['group_labels']\n"
         "pd.DataFrame({'configuration': [combo_name(c, labels) for c in feature_group_combinations()]})"),
    MD("## 2. Run the analysis (already executed; use the command line for the full run)\n"
       "```bash\npython scripts/run_nested_cv.py --run-name main\npython scripts/make_figures.py --run-name main\n```\n"
       "A quick end-to-end check: `python scripts/run_nested_cv.py --run-name smoke --smoke`."),
    MD("## 3. Headline results (pre-specified primary model: L2 logistic regression with class weighting)"),
    CODE("summ = pd.read_csv(RUN / 'tables' / 'metrics_summary.csv')\n"
         "pm, ps = cfg['models']['primary_model'], cfg['models']['primary_strategy']\n"
         "prim = summ[(summ.model == pm) & (summ.strategy == ps)]\n"
         "cols = ['config','roc_auc','roc_auc_ci_low','roc_auc_ci_high','pr_auc','sensitivity','specificity','precision','f1',\n"
         "        'balanced_accuracy','brier','calibration_slope','mean_n_features']\n"
         "prim[cols].round(3).reset_index(drop=True)"),
    MD("## 4. All models and imbalance strategies (sensitivity analysis)"),
    CODE("summ.pivot_table(index='config', columns=['model','strategy'], values='roc_auc').round(3)"),
    MD("## 5. Figures"),
    CODE("fig = RUN / 'figures'\n"
         "for f in ['roc_all_configs_%s_%s.png' % (pm, ps), 'pr_all_configs_%s_%s.png' % (pm, ps),\n"
         "          'calibration_all_configs_%s_%s.png' % (pm, ps), 'decision_curves_%s_%s.png' % (pm, ps),\n"
         "          'config_auc_ci_%s_%s.png' % (pm, ps), 'heatmap_auc_config_model_strategy.png',\n"
         "          'feature_selection_stability_heatmap.png']:\n"
         "    if (fig / f).exists():\n"
         "        print(f); display(Image(filename=str(fig / f), width=800))"),
    MD("## 6. Interpretation guardrails\n"
       "* Confidence intervals are patient-bootstrap intervals of the out-of-fold predictions; they ignore variability of the "
       "training procedure and are optimistic.\n"
       "* The configuration with the highest AUC is selected *after* looking at the same data – its AUC is optimistically biased.\n"
       "* Pairwise comparisons are descriptive; no multiplicity-adjusted significance claim is supported.\n"
       "* 27 events and 2–5 selected features: results are hypothesis-generating."),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python",
                                                                "name": "python3"}})
out = os.path.join(ROOT, "notebooks", "thyroid_improved.ipynb")
nbf.write(nb, out)
print("wrote", out)
