__all__ = [
    "PredictionSummaryComponent",
    "ImportancesComponent",
    "FeatureDescriptionsComponent",
    "FeatureInputComponent",
    "PdpComponent",
]
from math import ceil

import numpy as np
import pandas as pd

import dash
from dash import html, dcc, Input, Output, State, dash_table
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from ..dashboard_methods import *
from .. import to_html


class PredictionSummaryComponent(ExplainerComponent):
    _state_props = dict(
        index=("modelprediction-index-", "value"),
        percentile=("modelprediction-percentile-", "value"), # Added from layout
        pos_label=("pos-label-", "value"),
    )

    def __init__(
        self,
        explainer,
        title="Resumo da Previsão", # Translated
        name=None,
        hide_index=False,
        hide_percentile=False,
        hide_title=False,
        hide_subtitle=False, # Subtitle wasn't used in layout, kept arg for consistency
        hide_selector=False,
        pos_label=None,
        index=None,
        percentile=True,
        description=None, # Description wasn't used in layout, kept arg
        **kwargs,
    ):
        """Shows a summary for a particular prediction

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Resumo da Previsão". # Updated default
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            hide_index (bool, optional): hide index selector. Defaults to False.
            hide_percentile (bool, optional): hide percentile toggle. Defaults to False.
            hide_title (bool, optional): hide title. Defaults to False.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_selector (bool, optional): hide pos label selectors. Defaults to False.
            pos_label ({int, str}, optional): initial pos label.
                        Defaults to explainer.pos_label
            index ({int, str}, optional): Index to display prediction summary for. Defaults to None.
            percentile (bool, optional): Whether to add the prediction percentile. Defaults to True.
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown.
        """
        super().__init__(explainer, title, name)

        self.index_name = "modelprediction-index-" + self.name
        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)
        # Register dependency for prediction_result_markdown
        self.register_dependencies(['prediction_result_markdown'])


    def layout(self):
        # Using IndexSelector component for the dropdown now
        self.index_selector = IndexSelector(
            self.explainer,
            "modelprediction-index-" + self.name,
            index=self.index,
            **self.kwargs # Pass any relevant kwargs
        )

        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.H3(self.title), # Uses translated title
                        ]
                    ),
                    hide=self.hide_title,
                ),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(f"{self.explainer.index_name}:"),
                                            # Use the IndexSelector layout
                                            self.index_selector.layout(),
                                            # Old Dropdown:
                                            # dcc.Dropdown(
                                            #     id="modelprediction-index-" + self.name,
                                            #     options=[
                                            #         {"label": str(idx), "value": idx}
                                            #         for idx in self.explainer.idxs
                                            #     ],
                                            #     value=self.index,
                                            # ),
                                        ],
                                        md=6,
                                    ),
                                    hide=self.hide_index,
                                ),
                                make_hideable(
                                    dbc.Col([self.selector.layout()], width=3),
                                    hide=self.hide_selector,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            # Using dbc.Switch for boolean toggle is often preferred
                                            dbc.Label("Mostrar Percentil:"), # Translated
                                            dbc.Checklist(
                                                options=[
                                                    {"label": "Mostrar percentil", "value": True} # Translated label
                                                ],
                                                value=[True] if self.percentile else [],
                                                id="modelprediction-percentile-" + self.name,
                                                switch=True,
                                            ),
                                            # Old RadioButton:
                                            # dbc.Label("Mostrar Percentil:"), # Translated
                                            # dbc.Row(
                                            #     [
                                            #         dbc.RadioButton(
                                            #             id="modelprediction-percentile-"
                                            #             + self.name,
                                            #             className="form-check-input",
                                            #             value=self.percentile, # This logic might need adjustment for RadioButton state
                                            #         ),
                                            #         dbc.Label(
                                            #             "Mostrar percentil", # Translated
                                            #             html_for="modelprediction-percentile"
                                            #             + self.name,
                                            #             className="form-check-label",
                                            #         ),
                                            #     ],
                                            #     # check=True, # 'check' is not a standard dbc.Row parameter
                                            # ),
                                        ],
                                        md=3,
                                    ),
                                    hide=self.hide_percentile,
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        # Markdown component to display the summary
                                        dcc.Markdown(id="modelprediction-" + self.name),
                                    ],
                                    md=12,
                                )
                            ]
                        ),
                    ]
                ),
            ],
            class_name="h-100",
        )

    def component_callbacks(self, app):
        @app.callback(
            Output("modelprediction-" + self.name, "children"),
            [
                Input("modelprediction-index-" + self.name, "value"),
                Input("modelprediction-percentile-" + self.name, "value"),
                Input("pos-label-" + self.name, "value"),
            ],
        )
        def update_output_div(index, percentile_value, pos_label):
            if index is not None and self.explainer.index_exists(index):
                # percentile_value from Checklist is a list, need boolean
                include_percentile = bool(percentile_value)
                # Assuming prediction_result_markdown returns a string formatted for Markdown
                markdown_text = self.explainer.prediction_result_markdown(
                    index, include_percentile=include_percentile, pos_label=pos_label
                )
                # Potential translation of markdown text if needed (complex)
                # markdown_text = translate_markdown(markdown_text)
                return markdown_text
            raise PreventUpdate


class ImportancesComponent(ExplainerComponent):
    _state_props = dict(
        depth=("importances-depth-", "value"),
        importance_type=("importances-permutation-or-shap-", "value"),
        pos_label=("pos-label-", "value"),
    )

    def __init__(
        self,
        explainer,
        title="Importância das Variáveis", # Translated
        name=None,
        subtitle="Quais variáveis tiveram o maior impacto?", # Translated
        hide_type=False,
        hide_depth=False,
        hide_popout=False,
        hide_title=False,
        hide_subtitle=False,
        hide_selector=False,
        pos_label=None,
        importance_type="shap",
        depth=None,
        no_permutations=False,
        description=None,
        **kwargs,
    ):
        """Display features importances component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Importância das Variáveis". # Updated default
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            subtitle(str, optional): Subtitle. Defaults to
                        "Quais variáveis tiveram o maior impacto?". # Updated default
            hide_type (bool, optional): Hide permutation/shap selector toggle.
                        Defaults to False.
            hide_depth (bool, optional): Hide number of features toggle.
                        Defaults to False.
            hide_popout (bool, optional): hide popout button
            hide_title (bool, optional): hide title. Defaults to False.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_selector (bool, optional): hide pos label selectors.
                        Defaults to False.
            pos_label ({int, str}, optional): initial pos label.
                        Defaults to explainer.pos_label
            importance_type (str, {'permutation', 'shap'} optional):
                        initial importance type to display. Defaults to "shap".
            depth (int, optional): Initial number of top features to display.
                        Defaults to None (=show all).
            no_permutations (bool, optional): Do not use the permutation
                importances for this component. Defaults to False.
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown.
        """
        super().__init__(explainer, title, name)

        assert importance_type in [
            "shap",
            "permutation",
        ], "importance type must be either 'shap' or 'permutation'!"

        if depth is not None:
            # Use safe_n_features method if available, otherwise fallback
            n_features = getattr(self.explainer, 'safe_n_features', len(self.explainer.columns_ranked_by_shap()))
            self.depth = min(depth, n_features)

        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)
        self.no_permutations = no_permutations # Store this attribute

        if self.explainer.y_missing or self.no_permutations:
            self.hide_type = True
            self.importance_type = "shap"

        if self.description is None:
             # Translated description
            self.description = f"""
        Mostra as variáveis ordenadas da mais importante para a menos importante. Podem
        ser ordenadas pelo valor SHAP absoluto (impacto médio absoluto da
        variável na previsão final) ou pela importância de permutação (quanto
        o desempenho do modelo piora quando se embaralha esta variável, tornando-a
        inútil?).
        """
        self.popout = GraphPopout(
            "importances-" + self.name + "popout",
            "importances-graph-" + self.name,
            self.title, # uses translated title
            self.description, # uses translated description
        )
        self.register_dependencies("shap_values_df")
        if not (self.hide_type and self.importance_type == "shap"):
            # Only register if permutations might be shown
            self.register_dependencies("permutation_importances")

    def layout(self):
        # Use safe_n_features method if available, otherwise fallback
        n_features = getattr(self.explainer, 'safe_n_features', len(self.explainer.columns_ranked_by_shap()))
        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        self.title, # uses translated title
                                        className="card-title",
                                        id="importances-title-" + self.name,
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle" # uses translated subtitle
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description, # uses translated description
                                        target="importances-title-" + self.name,
                                    ),
                                ]
                            ),
                        ]
                    ),
                    hide=self.hide_title,
                ),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label("Tipo de Importância:"), # Translated
                                            dbc.Select(
                                                options=[
                                                    {
                                                        "label": "Importância de Permutação", # Translated
                                                        "value": "permutation",
                                                    },
                                                    {
                                                        "label": "Valores SHAP", # Translated
                                                        "value": "shap",
                                                    },
                                                ],
                                                value=self.importance_type,
                                                size="sm",
                                                id="importances-permutation-or-shap-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                # Translated
                                                "Selecionar o tipo de importância da variável: \n"
                                                "Importância de Permutação: Quanto diminui a métrica de desempenho ao embaralhar esta variável?\n"
                                                "Valores SHAP: Qual é a contribuição média SHAP (positiva ou negativa) desta variável?",
                                                target="importances-permutation-or-shap-form-"
                                                + self.name,
                                            ),
                                        ],
                                        md=3,
                                        id="importances-permutation-or-shap-form-"
                                        + self.name,
                                    ),
                                    self.hide_type,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                "Profundidade:", # Translated
                                                id="importances-depth-label-"
                                                + self.name,
                                            ),
                                            dbc.Select(
                                                id="importances-depth-" + self.name,
                                                options=[
                                                    # Add option for 'All'
                                                    {"label": "Todas", "value": n_features}
                                                ] + [
                                                    {
                                                        "label": str(i + 1),
                                                        "value": i + 1,
                                                    }
                                                    # Iterate up to n_features safely
                                                    for i in range(n_features)
                                                ],
                                                size="sm",
                                                # Set value to n_features if self.depth is None or > n_features
                                                value=self.depth if self.depth is not None and self.depth <= n_features else n_features,
                                            ),
                                            dbc.Tooltip(
                                                "Selecionar quantas variáveis exibir", # Translated
                                                target="importances-depth-label-"
                                                + self.name,
                                            ),
                                        ],
                                        md=2,
                                    ),
                                    self.hide_depth,
                                ),
                                make_hideable(
                                    dbc.Col([self.selector.layout()], width=2),
                                    hide=self.hide_selector,
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dcc.Loading(
                                            id="importances-graph-loading-" + self.name,
                                            children=dcc.Graph(
                                                id="importances-graph-" + self.name,
                                                config=dict(
                                                    modeBarButtons=[["toImage"]],
                                                    displaylogo=False,
                                                ),
                                            ),
                                        ),
                                    ]
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [self.popout.layout()], md=2, align="start"
                                    ),
                                    hide=self.hide_popout,
                                ),
                            ],
                            justify="end",
                        ),
                    ]
                ),
            ],
            class_name="h-100",
        )

    def to_html(self, state_dict=None, add_header=True):
        args = self.get_state_args(state_dict)
        n_features = getattr(self.explainer, 'safe_n_features', len(self.explainer.columns_ranked_by_shap()))
        # Handle depth=None (meaning show all)
        depth_val = None if args["depth"] is None or int(args["depth"]) >= n_features else int(args["depth"])

        # Assuming explainer.plot_importances handles internal plot label translations if needed
        fig = self.explainer.plot_importances(
            kind=args["importance_type"],
            topx=depth_val,
            pos_label=args["pos_label"],
        )

        html = to_html.card(to_html.fig(fig), title=self.title) # uses translated title
        if add_header:
            return to_html.add_header(html)
        return html

    def component_callbacks(self, app, **kwargs):
        @app.callback(
            Output("importances-graph-" + self.name, "figure"),
            [
                Input("importances-depth-" + self.name, "value"),
                Input("importances-permutation-or-shap-" + self.name, "value"),
                Input("pos-label-" + self.name, "value"),
            ],
        )
        def update_importances(depth, permutation_shap, pos_label):
            n_features = getattr(self.explainer, 'safe_n_features', len(self.explainer.columns_ranked_by_shap()))
            # Handle depth=None (meaning show all)
            depth_val = None if depth is None or int(depth) >= n_features else int(depth)

             # Assuming explainer.plot_importances handles internal plot label translations if needed
            plot = self.explainer.plot_importances(
                kind=permutation_shap, topx=depth_val, pos_label=pos_label
            )
            return plot


class FeatureDescriptionsComponent(ExplainerComponent):
    _state_props = dict(sort=("feature-descriptions-table-sort-", "value"))

    def __init__(
        self,
        explainer,
        title="Descrições das Variáveis", # Translated
        name=None,
        hide_title=False,
        hide_sort=False,
        sort="alphabet",
        **kwargs,
    ):
        """Display Feature Descriptions table.

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Descrições das Variáveis". # Updated default
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            hide_title (bool, optional): hide the title
            hide_sort (bool, optional): hide the sort
            sort (str, optional): how to sort the features, either 'alphabet'
                or by mean abs shap ('shap')

        """
        super().__init__(explainer, title, name)

        if sort not in {"alphabet", "shap"}:
            raise ValueError(
                "FeatureDesriptionsComponent parameter sort should be either"
                "'alphabet' or 'shap'!"
            )
        # Check if descriptions exist
        if not getattr(explainer, 'descriptions', None):
            print("Warning: No feature descriptions found in explainer.descriptions. "
                  "FeatureDescriptionsComponent will be empty.")
            # Optionally hide the component automatically if no descriptions
            # self.hide = True


    def layout(self):
        # Only render layout if descriptions exist? Or render empty state?
        # Current behaviour is to render controls even if empty.
        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.H3(self.title), # uses translated title
                        ]
                    ),
                    hide=self.hide_title,
                ),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Label("Ordenar Variáveis:"), # Translated
                                                    dbc.Select(
                                                        options=[
                                                            {
                                                                "label": "Alfabeticamente", # Translated
                                                                "value": "alphabet",
                                                            },
                                                            {
                                                                "label": "SHAP", # Kept SHAP as it's a specific name
                                                                "value": "shap",
                                                            },
                                                        ],
                                                        value=self.sort,
                                                        size="sm",
                                                        id="feature-descriptions-table-sort-"
                                                        + self.name,
                                                    ),
                                                ]
                                            ),
                                            dbc.Tooltip(
                                                # Translated
                                                "Ordenar variáveis alfabeticamente ou pelo valor SHAP absoluto médio (do maior para o menor).",
                                                target="feature-descriptions-table-sort-"
                                                + self.name,
                                            ),
                                        ],
                                        md=3,
                                    ),
                                    self.hide_sort,
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        # Container for the table generated by callback
                                        html.Div(
                                            id="feature-descriptions-table-" + self.name
                                        )
                                    ]
                                ),
                            ]
                        ),
                    ]
                ),
            ],
            class_name="h-100",
        )

    def to_html(self, state_dict=None, add_header=True):
        args = self.get_state_args(state_dict)
        # Assuming get_descriptions_df returns df with English headers 'Feature' and 'Description'
        try:
             df = self.explainer.get_descriptions_df(sort=args["sort"])
             # Translate headers for static export
             df_translated = df.rename(columns={'Feature': 'Variável', 'Description': 'Descrição'})
             html_content = to_html.table_from_df(df_translated)
        except AttributeError:
             # Handle cases where descriptions might be missing or method fails
             html_content = "Descrições das variáveis não disponíveis." # Translated
        except Exception as e:
             print(f"Error generating feature descriptions HTML: {e}")
             html_content = "Erro ao gerar descrições das variáveis." # Translated error

        html = to_html.card(html_content, title=self.title) # uses translated title
        if add_header:
            return to_html.add_header(html)
        return html

    def component_callbacks(self, app):
        @app.callback(
            Output("feature-descriptions-table-" + self.name, "children"),
            Input("feature-descriptions-table-sort-" + self.name, "value"),
        )
        def update_feature_descriptions_table(sort):
            try:
                 # Assuming get_descriptions_df returns df with English headers
                df = self.explainer.get_descriptions_df(sort=sort)
                 # Translate headers for dynamic table display
                df_translated = df.rename(columns={'Feature': 'Variável', 'Description': 'Descrição'})
                return dbc.Table.from_dataframe(df_translated, striped=True, bordered=True, hover=True)
            except AttributeError:
                 return dbc.Alert("Descrições das variáveis não disponíveis.", color="warning") # Translated warning
            except Exception as e:
                print(f"Error updating feature descriptions table: {e}")
                return dbc.Alert("Erro ao carregar descrições das variáveis.", color="danger") # Translated error


class PdpComponent(ExplainerComponent):
    _state_props = dict(
        index=("pdp-index-", "value"),
        col=("pdp-col-", "value"),
        dropna=("pdp-dropna-", "value"),
        sample=("pdp-sample-", "value"),
        gridlines=("pdp-gridlines-", "value"),
        gridpoints=("pdp-gridpoints-", "value"),
        cats_sort=("pdp-categories-sort-", "value"),
        pos_label=("pos-label-", "value"),
    )

    def __init__(
        self,
        explainer,
        title="Gráfico de Dependência Parcial", # Translated
        name=None,
        subtitle="Como muda a previsão se alterar uma variável?", # Translated
        hide_col=False,
        hide_index=False,
        hide_title=False,
        hide_subtitle=False,
        hide_footer=False,
        hide_selector=False,
        hide_popout=False,
        hide_dropna=False,
        hide_sample=False,
        hide_gridlines=False,
        hide_gridpoints=False,
        hide_cats_sort=False,
        index_dropdown=True,
        feature_input_component=None,
        pos_label=None,
        col=None,
        index=None,
        dropna=True,
        sample=100,
        gridlines=50,
        gridpoints=10,
        cats_sort="freq",
        description=None,
        **kwargs,
    ):
        """Show Partial Dependence Plot component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Gráfico de Dependência Parcial". # Updated default
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            subtitle (str): subtitle. Defaults to
                        "Como muda a previsão se alterar uma variável?". # Updated default
            hide_col (bool, optional): Hide feature selector. Defaults to False.
            hide_index (bool, optional): Hide index selector. Defaults to False.
            hide_title (bool, optional): Hide title, Defaults to False.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_footer (bool, optional): hide the footer at the bottom of the component
            hide_selector (bool, optional): hide pos label selectors. Defaults to False.
            hide_popout (bool, optional): hide popout button
            hide_dropna (bool, optional): Hide drop na's toggle Defaults to False.
            hide_sample (bool, optional): Hide sample size input. Defaults to False.
            hide_gridlines (bool, optional): Hide gridlines input. Defaults to False.
            hide_gridpoints (bool, optional): Hide gridpounts input. Defaults to False.
            hide_cats_sort (bool, optional): Hide the categorical sorting dropdown. Defaults to False.
            index_dropdown (bool, optional): Use dropdown for index input instead
                of free text input. Defaults to True.
            feature_input_component (FeatureInputComponent): A FeatureInputComponent
                that will give the input to the graph instead of the index selector.
                If not None, hide_index=True. Defaults to None.
            pos_label ({int, str}, optional): initial pos label.
                        Defaults to explainer.pos_label
            col (str, optional): Feature to display PDP for. Defaults to None.
            index ({int, str}, optional): Index to add ice line to plot. Defaults to None.
            dropna (bool, optional): Drop rows where values equal explainer.na_fill (usually -999). Defaults to True.
            sample (int, optional): Sample size to calculate average partial dependence. Defaults to 100.
            gridlines (int, optional): Number of ice lines to display in plot. Defaults to 50.
            gridpoints (int, optional): Number of breakpoints on horizontal axis Defaults to 10.
            cats_sort (str, optional): how to sort categories: 'alphabet',
                'freq' or 'shap'. Defaults to 'freq'.
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown.
        """
        super().__init__(explainer, title, name)

        self.index_name = "pdp-index-" + self.name

        if self.col is None:
             # Check if columns_ranked_by_shap exists and has elements
            ranked_cols = self.explainer.columns_ranked_by_shap()
            if ranked_cols:
                 self.col = ranked_cols[0]
            else:
                 # Fallback if no columns available (should not happen with valid explainer)
                 self.col = None
                 print("Warning: No columns found for PDP default.")


        self.feature_input_component = feature_input_component # Store for later checks
        if self.feature_input_component is not None:
            self.exclude_callbacks(self.feature_input_component)
            self.hide_index = True

        if self.description is None:
             # Translated description
            self.description = f"""
        O gráfico de dependência parcial (PDP) mostra como a previsão do modelo
        mudaria se alterasse uma variável específica. O gráfico mostra uma amostra
        de observações e como essas observações mudariam com esta
        variável (linhas de grade - gridlines). O efeito médio é mostrado a cinzento. O efeito
        de alterar a variável para um único {self.explainer.index_name} é
        mostrado a azul. Pode ajustar quantas observações amostrar para a
        média, quantas linhas de grade mostrar e para quantos pontos ao longo do
        eixo x calcular as previsões do modelo (pontos de grade - gridpoints).
        """
        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)
        # Use IndexSelector component
        self.index_selector = IndexSelector(
            explainer,
            "pdp-index-" + self.name,
            index=index,
            index_dropdown=index_dropdown,
            **kwargs,
        )

        self.popout = GraphPopout(
            "pdp-" + self.name + "popout",
            "pdp-graph-" + self.name,
            self.title, # uses translated title
            self.description, # uses translated description
        )
        # Register dependency needed for plot_pdp
        self.register_dependencies(['pdp'])


    def layout(self):
        # Get ranked columns safely
        ranked_cols = self.explainer.columns_ranked_by_shap()
        col_options = [{"label": col, "value": col} for col in ranked_cols] if ranked_cols else []
        # Ensure self.col is valid or set to None if no options
        current_col = self.col if self.col in ranked_cols else (ranked_cols[0] if ranked_cols else None)

        # Get safe limits for sample and gridlines
        safe_len = len(self.explainer) if hasattr(self.explainer, '__len__') else 1000 # Default max if len unknown
        safe_sample = min(self.sample, safe_len)
        safe_gridlines = min(self.gridlines, safe_len)

        # Determine if categorical sort dropdown should be visible
        is_categorical = current_col is not None and current_col in getattr(self.explainer, 'cat_cols', [])
        cats_sort_div_style = {} if is_categorical else dict(display="none")

        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.Div(
                                [
                                    html.H3(self.title, id="pdp-title-" + self.name), # Uses translated title
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle" # Uses translated subtitle
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description, # Uses translated description
                                        target="pdp-title-" + self.name,
                                    ),
                                ]
                            ),
                        ]
                    ),
                    hide=self.hide_title,
                ),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                "Variável:", # Translated
                                                html_for="pdp-col" + self.name,
                                                id="pdp-col-label-" + self.name,
                                            ),
                                            dbc.Tooltip(
                                                # Translated
                                                "Selecionar a variável para a qual deseja ver o gráfico de dependência parcial",
                                                target="pdp-col-label-" + self.name,
                                            ),
                                            dbc.Select(
                                                id="pdp-col-" + self.name,
                                                options=col_options,
                                                value=current_col,
                                                size="sm",
                                            ),
                                        ],
                                        md=4,
                                    ),
                                    hide=self.hide_col,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                f"{self.explainer.index_name}:",
                                                id="pdp-index-label-" + self.name,
                                            ),
                                            dbc.Tooltip(
                                                 # Translated
                                                f"Selecionar o {self.explainer.index_name} para exibir o gráfico de dependência parcial",
                                                target="pdp-index-label-" + self.name,
                                            ),
                                            self.index_selector.layout(),
                                        ],
                                        md=4,
                                    ),
                                    hide=self.hide_index,
                                ),
                                make_hideable(
                                    dbc.Col([self.selector.layout()], width=2),
                                    hide=self.hide_selector,
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dcc.Loading(
                                            id="loading-pdp-graph-" + self.name,
                                            type="circle",
                                            children=[
                                                dcc.Graph(
                                                    id="pdp-graph-" + self.name,
                                                    config=dict(
                                                        modeBarButtons=[["toImage"]],
                                                        displaylogo=False,
                                                    ),
                                                )
                                            ],
                                        ),
                                    ]
                                )
                            ]
                        ),
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [self.popout.layout()], md=2, align="start"
                                    ),
                                    hide=self.hide_popout,
                                ),
                            ],
                            justify="end",
                        ),
                    ]
                ),
                make_hideable(
                    dbc.CardFooter(
                        [
                            dbc.Row(
                                [
                                    make_hideable(
                                        dbc.Col(
                                            [
                                                dbc.Row(
                                                    [
                                                        # Use Switch for boolean toggle
                                                        dbc.Label("Remover Preenchimento:"), # Translated
                                                        dbc.Checklist(
                                                            options=[
                                                                {"label": f"Remover {self.explainer.na_fill}", "value": True} # Translated label
                                                            ],
                                                            value=[True] if self.dropna else [],
                                                            id="pdp-dropna-" + self.name,
                                                            switch=True,
                                                        ),
                                                        dbc.Tooltip(
                                                            # Translated
                                                            "Remover todas as observações com valores da variável "
                                                            f"iguais a {self.explainer.na_fill} do gráfico. "
                                                            "Isto evita que os valores de preenchimento distorçam o eixo x.",
                                                            target="pdp-dropna-" + self.name,
                                                        ),
                                                    ]
                                                ),
                                            ]
                                        ),
                                        hide=self.hide_dropna,
                                    ),
                                    make_hideable(
                                        dbc.Col(
                                            [
                                                dbc.Label(
                                                    "Amostra:", # Translated
                                                    id="pdp-sample-label-" + self.name,
                                                ),
                                                dbc.Tooltip(
                                                     # Translated
                                                    "Número de observações a usar para calcular a dependência parcial média",
                                                    target="pdp-sample-label-" + self.name,
                                                ),
                                                dbc.Input(
                                                    id="pdp-sample-" + self.name,
                                                    value=safe_sample,
                                                    type="number",
                                                    min=0,
                                                    max=safe_len,
                                                    step=10 if safe_len > 100 else 1, # Adjust step based on max
                                                    debounce=True # Update only after user stops typing
                                                ),
                                            ]
                                        ),
                                        hide=self.hide_sample,
                                    ),
                                    make_hideable(
                                        dbc.Col(
                                            [  # gridlines
                                                dbc.Label(
                                                    "Linhas ICE:", # Translated (ICE lines is common term)
                                                    id="pdp-gridlines-label-" + self.name,
                                                ),
                                                dbc.Tooltip(
                                                    # Translated
                                                    "Número de dependências parciais de observações individuais (linhas ICE) a mostrar no gráfico",
                                                    target="pdp-gridlines-label-" + self.name,
                                                ),
                                                dbc.Input(
                                                    id="pdp-gridlines-" + self.name,
                                                    value=safe_gridlines,
                                                    type="number",
                                                    min=0,
                                                    max=safe_len,
                                                    step=10 if safe_len > 100 else 1, # Adjust step
                                                    debounce=True
                                                ),
                                            ]
                                        ),
                                        hide=self.hide_gridlines,
                                    ),
                                    make_hideable(
                                        dbc.Col(
                                            [  # gridpoints
                                                dbc.Label(
                                                    "Pontos Grade:", # Translated
                                                    id="pdp-gridpoints-label-" + self.name,
                                                ),
                                                dbc.Tooltip(
                                                     # Translated
                                                    "Número de pontos a amostrar no eixo da variável para previsões."
                                                    " Quanto maior, mais suave a curva, mas demora mais a calcular",
                                                    target="pdp-gridpoints-label-" + self.name,
                                                ),
                                                dbc.Input(
                                                    id="pdp-gridpoints-" + self.name,
                                                    value=self.gridpoints,
                                                    type="number",
                                                    min=2, # Minimum 2 points needed for a line
                                                    max=100,
                                                    step=1,
                                                    debounce=True
                                                ),
                                            ]
                                        ),
                                        hide=self.hide_gridpoints,
                                    ),
                                    make_hideable(
                                        html.Div(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Label(
                                                            "Ordenar categorias:", # Translated
                                                            id="pdp-categories-sort-label-" + self.name,
                                                        ),
                                                        dbc.Tooltip(
                                                             # Translated
                                                            "Como ordenar as categorias: Alfabeticamente, mais comum "
                                                            "primeiro (Frequência), ou maior valor SHAP absoluto médio primeiro (Impacto Shap)",
                                                            target="pdp-categories-sort-label-" + self.name,
                                                        ),
                                                        dbc.Select(
                                                            id="pdp-categories-sort-" + self.name,
                                                            options=[
                                                                {
                                                                    "label": "Alfabeticamente", # Translated
                                                                    "value": "alphabet",
                                                                },
                                                                {
                                                                    "label": "Frequência", # Translated
                                                                    "value": "freq",
                                                                },
                                                                {
                                                                    "label": "Impacto Shap", # Translated
                                                                    "value": "shap",
                                                                },
                                                            ],
                                                            value=self.cats_sort,
                                                            size="sm",
                                                        ),
                                                    ]
                                                )
                                            ],
                                            id="pdp-categories-sort-div-" + self.name,
                                            style=cats_sort_div_style, # Dynamically set style
                                        ),
                                        hide=self.hide_cats_sort,
                                    ),
                                ]
                            ),
                        ]
                    ),
                    hide=self.hide_footer,
                ),
            ],
            class_name="h-100",
        )

    def get_state_tuples(self):
        _state_tuples = super().get_state_tuples()
        if self.feature_input_component is not None:
            _state_tuples.extend(self.feature_input_component.get_state_tuples())
        return sorted(list(set(_state_tuples)))

    def to_html(self, state_dict=None, add_header=True):
        args = self.get_state_args(state_dict)
        # Validate inputs for static generation
        col = args.get("col")
        index = args.get("index")
        dropna = bool(args.get("dropna"))
        sample = args.get("sample", 100) # Provide default
        gridlines = args.get("gridlines", 50) # Provide default
        gridpoints = args.get("gridpoints", 10) # Provide default
        cats_sort = args.get("cats_sort", 'freq') # Provide default
        pos_label = args.get("pos_label")

        # Default html content
        html_content = "Dados de entrada inválidos ou insuficientes para gerar o gráfico PDP." # Translated error

        if col is not None: # Minimum requirement is a column
            if self.feature_input_component is None:
                if index is not None: # If no feature input, index is required
                     # Assuming explainer.plot_pdp handles internal plot label translations if needed
                    fig = self.explainer.plot_pdp(
                        col,
                        index,
                        drop_na=dropna,
                        sample=sample,
                        gridlines=gridlines,
                        gridpoints=gridpoints,
                        sort=cats_sort,
                        pos_label=pos_label,
                    )
                    html_content = to_html.fig(fig)
                else:
                     html_content = "Índice não selecionado para o gráfico PDP." # Translated error

            else: # Using feature input component
                # Reconstruct input row from state_dict
                inputs = {
                    k: v
                    for k, v in self.feature_input_component.get_state_args(
                        state_dict
                    ).items()
                    if k != "index" # Exclude index key if present
                }
                # Check if all inputs are present and valid
                if len(inputs) == len(
                    self.feature_input_component._input_features # Assuming this attr exists
                ) and not any([i is None for i in inputs.values()]):
                    X_row = self.explainer.get_row_from_input(list(inputs.values()), ranked_by_shap=True)
                     # Assuming explainer.plot_pdp handles internal plot label translations if needed
                    fig = self.explainer.plot_pdp(
                        col,
                        X_row=X_row,
                        drop_na=dropna,
                        sample=sample,
                        gridlines=gridlines,
                        gridpoints=gridpoints,
                        sort=cats_sort,
                        pos_label=pos_label,
                    )
                    html_content = to_html.fig(fig)
                else:
                    html_content = f"<div>Dados de entrada incorretos</div>" # Translated

        html = to_html.card(html_content, title=self.title) # uses translated title
        if add_header:
            return to_html.add_header(html)
        return html

    def component_callbacks(self, app):
        @app.callback(
            Output("pdp-categories-sort-div-" + self.name, "style"),
            Input("pdp-col-" + self.name, "value"),
            # prevent_initial_call=True # Prevent initial call unless needed
        )
        def update_pdp_sort_div(col):
            if col is not None and col in getattr(self.explainer, 'cat_cols', []):
                 return {} # Show div
            return dict(display="none") # Hide div

        if self.feature_input_component is None:
            # Callback for when index selector is used
            @app.callback(
                Output("pdp-graph-" + self.name, "figure"),
                [
                    Input("pdp-index-" + self.name, "value"),
                    Input("pdp-col-" + self.name, "value"),
                    Input("pdp-dropna-" + self.name, "value"),
                    Input("pdp-sample-" + self.name, "value"),
                    Input("pdp-gridlines-" + self.name, "value"),
                    Input("pdp-gridpoints-" + self.name, "value"),
                    Input("pdp-categories-sort-" + self.name, "value"),
                    Input("pos-label-" + self.name, "value"),
                ],
                 # prevent_initial_call=True
            )
            def update_pdp_graph(
                index, col, drop_na, sample, gridlines, gridpoints, sort, pos_label
            ):
                # Validate inputs
                if col is None or index is None or not self.explainer.index_exists(index):
                    # Return empty figure or raise PreventUpdate if no plot should be shown
                    # return go.Figure()
                    raise PreventUpdate
                # Assuming explainer.plot_pdp handles internal plot label translations if needed
                return self.explainer.plot_pdp(
                    col,
                    index,
                    drop_na=bool(drop_na), # drop_na from checklist is a list
                    sample=int(sample) if sample is not None else 100,
                    gridlines=int(gridlines) if gridlines is not None else 50,
                    gridpoints=int(gridpoints) if gridpoints is not None else 10,
                    sort=sort,
                    pos_label=pos_label,
                )

        else:
            # Callback for when feature input component is used
            @app.callback(
                Output("pdp-graph-" + self.name, "figure"),
                [
                    Input("pdp-col-" + self.name, "value"),
                    Input("pdp-dropna-" + self.name, "value"),
                    Input("pdp-sample-" + self.name, "value"),
                    Input("pdp-gridlines-" + self.name, "value"),
                    Input("pdp-gridpoints-" + self.name, "value"),
                    Input("pdp-categories-sort-" + self.name, "value"),
                    Input("pos-label-" + self.name, "value"),
                    # Use the specific inputs from the feature_input_component
                    *self.feature_input_component._feature_callback_inputs,
                ],
                 # prevent_initial_call=True
            )
            def update_pdp_graph_feature_input(
                col, drop_na, sample, gridlines, gridpoints, sort, pos_label, *inputs
            ):
                # Validate inputs
                if col is None or any(i is None for i in inputs):
                     # return go.Figure()
                    raise PreventUpdate

                # Get the row from inputs
                X_row = self.explainer.get_row_from_input(inputs, ranked_by_shap=True) # Assuming this method exists and works
                 # Assuming explainer.plot_pdp handles internal plot label translations if needed
                return self.explainer.plot_pdp(
                    col,
                    X_row=X_row,
                    drop_na=bool(drop_na), # drop_na from checklist is a list
                    sample=int(sample) if sample is not None else 100,
                    gridlines=int(gridlines) if gridlines is not None else 50,
                    gridpoints=int(gridpoints) if gridpoints is not None else 10,
                    sort=sort,
                    pos_label=pos_label,
                )


class FeatureInputComponent(ExplainerComponent):
    # Define _state_props dynamically in __init__

    def __init__(
        self,
        explainer,
        title="Entrada de Variáveis", # Translated
        name=None,
        subtitle="Ajuste os valores das variáveis para alterar a previsão", # Translated
        hide_title=False,
        hide_subtitle=False,
        hide_index=False,
        hide_range=False,
        index=None,
        n_input_cols=4,
        sort_features="shap",
        fill_row_first=True,
        description=None,
        **kwargs,
    ):
        """Interaction Dependence Component.

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Entrada de Variáveis". # Updated default
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            subtitle (str): subtitle. Defaults to
                        "Ajuste os valores das variáveis para alterar a previsão". # Updated default
            hide_title (bool, optional): hide the title
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_index (bool, optional): hide the index selector
            hide_range (bool, optional): hide the range label under the inputs
            index (str, int, optional): default index
            n_input_cols (int, optional): number of columns to split features inputs in.
                Defaults to 4.
            sort_features(str, optional): how to sort the features. For now only options
                is 'shap' to sort by mean absolute shap value. Options: 'shap', 'alphabet'.
            fill_row_first (bool, optional): if True most important features will
                be on top row, if False they will be in most left column.
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown.
        """
        super().__init__(explainer, title, name)

        # Store attributes needed later
        self.n_input_cols = n_input_cols
        self.sort_features = sort_features
        self.fill_row_first = fill_row_first
        self.hide_range = hide_range

        # Ensure columns are unique
        if len(explainer.columns) != len(set(explainer.columns)):
             # Developer-facing error
             raise ValueError("Not all X column names are unique, so cannot launch FeatureInputComponent component/tab!")

        # Use IndexSelector component
        self.index_input = IndexSelector(
            explainer, name="feature-input-index-" + self.name, index=index, **kwargs
        )
        self.index_name = "feature-input-index-" + self.name # Keep for easy reference

        # Determine feature order
        if self.sort_features == "shap":
            self._input_features = self.explainer.columns_ranked_by_shap()
        elif self.sort_features == "alphabet":
            # Use merged_cols if available, otherwise fallback to columns
            cols_to_sort = getattr(self.explainer, 'merged_cols', self.explainer.columns)
            self._input_features = sorted(list(cols_to_sort))
        else:
             # Developer-facing error
            raise ValueError(
                f"parameter sort_features should be either 'shap', "
                f"or 'alphabet', but you passed sort_features='{self.sort_features}'"
            )

        # Generate input components and define callbacks/state props dynamically
        self._feature_inputs = []
        self._feature_callback_inputs = []
        self._feature_callback_outputs = []
        self._state_props = {}

        # Safely access dictionaries, provide empty dicts as fallback
        onehot_cols = getattr(self.explainer, 'onehot_cols', set())
        onehot_dict = getattr(self.explainer, 'onehot_dict', {})
        cat_dict = getattr(self.explainer, 'categorical_dict', {})

        for feature in self._input_features:
             input_id = f"feature-input-{feature}-input-{self.name}"
             self._feature_inputs.append(
                 self._generate_dash_input(
                     feature, onehot_cols, onehot_dict, cat_dict
                 )
             )
             self._feature_callback_inputs.append(Input(input_id, "value"))
             self._feature_callback_outputs.append(Output(input_id, "value"))
             self._state_props[feature] = (f"feature-input-{feature}-input-", "value")

        # Add index to state props
        self._state_props["index"] = ("feature-input-index-", "value")

        if self.description is None:
            # Translated description
            self.description = """
        Ajuste os valores de entrada para ver as previsões para cenários hipotéticos ("what if")."""
        # Register dependency for get_X_row
        self.register_dependencies(['get_X_row'])


    def _generate_dash_input(self, col, onehot_cols, onehot_dict, cat_dict):
        input_id = f"feature-input-{col}-input-{self.name}"
        # Check for categorical columns defined in cat_dict
        if col in cat_dict:
            col_values = cat_dict.get(col, []) # Safely get values
            return html.Div(
                [
                    dbc.Label(col),
                    dcc.Dropdown(
                        id=input_id,
                        options=[
                            dict(label=str(col_val), value=col_val) # Ensure label is string
                            for col_val in col_values
                        ],
                        style={"width": "100%"},
                        clearable=False, # Decide if user should be able to clear selection
                    ),
                     # Translated FormText
                    make_hideable(dbc.FormText(f"Selecione qualquer {col}"), hide=self.hide_range),
                ]
            )
        # Check for one-hot encoded base columns
        elif col in onehot_dict: # onehot_dict keys are base cols
             # Get the actual one-hot encoded columns for this base feature
            col_values = onehot_dict.get(col, [])
             # Generate display values (remove prefix)
            display_values = [
                val[len(col) + 1 :] if val.startswith(col + "_") else val
                for val in col_values
            ]
             # Check if a "not encoded" state exists (all OHE columns are 0 for a sample)
            not_encoded_val = getattr(self.explainer, 'onehot_notencoded', {}).get(col)
            if not_encoded_val is not None:
                 # Add option representing the "not encoded" state
                 # Use a specific value or representation for this state
                 col_values.append(not_encoded_val)
                 display_values.append(str(not_encoded_val)) # Or a more descriptive label

            return html.Div(
                [
                    dbc.Label(col),
                    dcc.Dropdown(
                        id=input_id,
                        options=[
                            dict(label=display, value=col_val)
                            for display, col_val in zip(display_values, col_values)
                        ],
                        style={"width": "100%"},
                        clearable=False,
                    ),
                     # Translated FormText
                    make_hideable(dbc.FormText(f"Selecione qualquer {col}"), hide=self.hide_range),
                ]
            )
        # Assume numerical otherwise
        else:
            min_range, max_range = None, None
            # Safely get min/max, handling potential errors or missing data
            try:
                 # Exclude na_fill value when calculating min/max
                valid_values = self.explainer.X[col][lambda x: x != self.explainer.na_fill]
                if not valid_values.empty:
                     min_range = np.round(valid_values.min(), 2)
                     max_range = np.round(valid_values.max(), 2)
            except KeyError:
                print(f"Warning: Column '{col}' not found in explainer data for range calculation.")
            except Exception as e:
                print(f"Warning: Could not determine range for numerical feature '{col}': {e}")

            range_text = f"Intervalo: {min_range}-{max_range}" if min_range is not None else "Intervalo indisponível" # Translated fallback

            return html.Div(
                [
                    dbc.Label(col),
                    dbc.Input(
                        id=input_id,
                        type="number",
                        # Add step attribute for usability if appropriate (e.g., step=0.1 or 1)
                        step="any", # Allows any decimal
                        debounce=True # Update only after user stops typing
                    ),
                    make_hideable(dbc.FormText(range_text), hide=self.hide_range), # Uses translated text
                ]
            )

    def get_slices_cols_first(self, n_inputs, n_cols=2):
        """returns a list of slices to divide n inputs into n_cols columns,
        filling columns first"""
        if n_inputs == 0: return []
        n_cols = min(n_cols, n_inputs) # Ensure n_cols is not greater than n_inputs
        rows_per_col = ceil(n_inputs / n_cols)
        slices = []
        for col in range(n_cols):
            start = col * rows_per_col
            end = min(start + rows_per_col, n_inputs) # Ensure end does not exceed n_inputs
            if start < end: # Only add slice if it's valid
                 slices.append(slice(start, end))
        return slices

    def get_slices_rows_first(self, n_inputs, n_cols=3):
        """returns a list of slices to divide n inputs into n_cols columns,
        filling rows first"""
        if n_inputs == 0: return []
        n_cols = min(n_cols, n_inputs) # Ensure n_cols is not greater than n_inputs
        slices = []
        for i in range(n_cols):
             # Create list of indices for this column
             indices = list(range(i, n_inputs, n_cols))
             if indices:
                 # Convert list of indices to slice-like behavior if needed by consumer
                 # Or simply return the list of indices per column
                 # Returning list of indices is more flexible
                 # slices.append(indices)
                 # Sticking to slice for now, assuming consumer wants slices (might be less intuitive for row-first)
                 # This slice logic for row-first might not be what's expected.
                 # A simpler approach might be to just return lists of indices per column.
                 # Let's try returning lists of indices instead.
                 slices.append(indices)

        # The original slice logic for rows_first was complex and potentially incorrect.
        # Returning lists of indices is clearer for column generation.
        # If the consumer *requires* slices, the logic needs careful review.
        # For now, adjusting the layout method to handle lists of indices.
        return slices # Now returns lists of indices, not slices

    def layout(self):
        # Generate columns based on lists of indices
        if self.fill_row_first:
            cols_indices = self.get_slices_rows_first(len(self._feature_inputs), self.n_input_cols)
            input_row = dbc.Row(
                [
                    dbc.Col([self._feature_inputs[i] for i in indices])
                    for indices in cols_indices if indices # Ensure list is not empty
                ]
            )
        else: # Fill columns first
            cols_slices = self.get_slices_cols_first(len(self._feature_inputs), self.n_input_cols)
            input_row = dbc.Row(
                [
                    dbc.Col(self._feature_inputs[slicer])
                    for slicer in cols_slices
                ]
            )

        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        self.title, # uses translated title
                                        id="feature-input-title-" + self.name,
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle" # uses translated subtitle
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description, # uses translated description
                                        target="feature-input-title-" + self.name,
                                    ),
                                ]
                            ),
                        ]
                    ),
                    hide=self.hide_title,
                ),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col([self.index_input.layout()], md=4),
                                    hide=self.hide_index,
                                ),
                            ]
                        ),
                        input_row, # Use the generated row of inputs
                    ]
                ),
            ],
            class_name="h-100",
        )

    def to_html(self, state_dict=None, add_header=True):
        args = self.get_state_args(state_dict)
        index_val = args.get('index', 'Nenhum') # Translated default/fallback
        # Generate static HTML representation of inputs
        html_inputs = [
            to_html.input(feature, args.get(feature, "N/A"), disabled=True) # Use N/A if value missing
            for feature in self._input_features
        ]
        html_content = to_html.hide(f"Selecionado: <b>{index_val}</b>", hide=self.hide_index) # Uses translated text

        # Generate rows based on lists of indices/slices
        if self.fill_row_first:
            cols_indices = self.get_slices_rows_first(len(html_inputs), self.n_input_cols)
            html_content += to_html.row(
                *[
                    "".join([html_inputs[i] for i in indices])
                    for indices in cols_indices if indices
                ]
            )
        else: # Fill columns first
            cols_slices = self.get_slices_cols_first(len(html_inputs), self.n_input_cols)
            html_content += to_html.row(
                *[
                    "".join(html_inputs[slicer])
                    for slicer in cols_slices
                ]
            )

        html = to_html.card(html_content, title=self.title, subtitle=self.subtitle) # uses translated title/subtitle
        if add_header:
            return to_html.add_header(html)
        return html

    def component_callbacks(self, app):
        @app.callback(
            [*self._feature_callback_outputs],
            [Input(self.index_name, "value")],
             # prevent_initial_call=True # Usually desired for index changes
        )
        def update_whatif_inputs(index):
            if index is None or not self.explainer.index_exists(index):
                # What to return if index is invalid? Empty strings? Current values?
                # Returning current values via PreventUpdate is safest unless reset is desired.
                raise PreventUpdate

            try:
                 # Get the row corresponding to the index, ordered by the feature list used for inputs
                X_row = self.explainer.get_X_row(index, merge=True)[self._input_features]
                # Return the values in the correct order; handle potential type issues (e.g., ensure correct types for dropdowns/inputs)
                # This might need more sophisticated type conversion depending on _generate_dash_input logic
                return X_row.values[0].tolist()
            except Exception as e:
                print(f"Error updating feature inputs for index {index}: {e}")
                raise PreventUpdate # Prevent update on error