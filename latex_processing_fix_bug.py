import pandas as pd

# 1. Charger les résultats
df = pd.read_csv("csv_scores/injection_strategies_output.csv")

# 2. Agréger par (task, context) sur les seeds
df_avg = df.groupby(["task", "context"])[["macro_f1", "weighted_f1"]].mean().reset_index()

# 3. Convertir en pourcentage et arrondir
df_avg[["macro_f1", "weighted_f1"]] = df_avg[["macro_f1", "weighted_f1"]] * 100
df_avg = df_avg.round(2)

# 4. Renommer les colonnes
df_avg = df_avg.rename(columns={"macro_f1": "mF1", "weighted_f1": "wF1"})

# 5. Ordre des méthodes (à adapter à ton fichier CSV)
desired_order = sorted(df_avg["context"].unique())  # automatique, ou tu peux spécifier manuellement

df_avg["context"] = pd.Categorical(df_avg["context"], categories=desired_order, ordered=True)
df_avg = df_avg.sort_values("context")

# 6. Groupes de tâches
task_groups = {
    "Group 1": ["PubMed_20k_RCT", "csabstracts"],
    "Group 2": ["DeepRhole", "legal-eval", "scotus-category", "scotus-rhetorical_function", "scotus-steps"],
    "All Tasks": df_avg["task"].unique().tolist()
}

# 7. Formattage pour LaTeX
def format_task_name(task_name):
    return task_name.replace("_", "\\_")

# 8. Génération du tableau LaTeX
def generate_latex_table(group_name, task_order):
    df_filtered = df_avg[df_avg["task"].isin(task_order)].copy()
    df_filtered["task"] = pd.Categorical(df_filtered["task"], categories=task_order, ordered=True)
    df_filtered = df_filtered.sort_values("task")

    task_names = [format_task_name(task) for task in task_order]
    df_pivot = df_filtered.pivot(index="context", columns="task", values=["mF1", "wF1"])

    latex_table = f"\\begin{{table*}}[!htbp]\n\\centering\n"
    latex_table += f"\\resizebox{{\\textwidth}}{{!}}{{%\n\\begin{{tabular}}{{ l |" + " r r |" * len(task_order) + " }\n"
    latex_table += "\\toprule\n"

    # En-têtes
    header_line = "\\textbf{Method} "
    for task in task_names:
        header_line += f"& \\multicolumn{{2}}{{c|}}{{\\textbf{{{task}}}}} "
    header_line += "\\\\\n"

    sub_header_line = "& \\textbf{mF1} & \\textbf{wF1} " * len(task_order) + "\\\\\n"
    latex_table += header_line
    latex_table += "\\cmidrule(lr){{2-{}}} \n".format(2 * len(task_order) + 1)
    latex_table += sub_header_line
    latex_table += "\\midrule\n"

    for context in desired_order:
        if context in df_pivot.index:
            row = f"{context.replace('_', ' ')} "
            for task in task_order:
                try:
                    mf1 = df_pivot.loc[context, ("mF1", task)]
                    wf1 = df_pivot.loc[context, ("wF1", task)]
                    row += f"& {mf1:.2f} & {wf1:.2f} "
                except KeyError:
                    row += "& -- & -- "
            row += "\\\\\n"
            latex_table += row

    latex_table += "\\bottomrule\n\\end{tabular}}\n"
    latex_table += f"\\caption{{Performance en mF1 et wF1 pour {group_name}.}}\n"
    latex_table += f"\\label{{tab:results_{group_name.replace(' ', '_').lower()}}}\n"
    latex_table += "\\end{table*}\n"

    return latex_table

# 9. Générer les tableaux LaTeX
for group_name, tasks in task_groups.items():
    print(f"\n% === {group_name} ===\n")
    print(generate_latex_table(group_name, tasks))
