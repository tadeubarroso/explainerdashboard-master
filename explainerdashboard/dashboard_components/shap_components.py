__all__ = [
    "ShapSummaryComponent",
    "ShapDependenceComponent",
    "ShapSummaryDependenceConnector",
    "InteractionSummaryComponent",
    "InteractionDependenceComponent",
    "InteractionSummaryDependenceConnector",
    "ShapContributionsTableComponent",
    "ShapContributionsGraphComponent",
]


import dash
from dash import html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from ..dashboard_methods import *
from .. import to_html


class ShapSummaryComponent(ExplainerComponent):
    _state_props = dict(
        summary_type=("shap-summary-type-", "value"),
        depth=("shap-summary-depth-", "value"),
        index=("shap-summary-index-", "value"),
        pos_label=("pos-label-", "value"),
    )

    def __init__(
        self,
        explainer,
        title="Sumário Shap", # Traduzido
        name=None,
        subtitle="Ordenando características por valor shap", # Traduzido
        hide_title=False,
        hide_subtitle=False,
        hide_depth=False,
        hide_type=False,
        hide_index=False,
        hide_selector=False,
        hide_popout=False,
        pos_label=None,
        depth=None,
        summary_type="aggregate",
        max_cat_colors=5,
        index=None,
        plot_sample=None,
        description=None,
        **kwargs,
    ):
        """Componente que mostra o sumário shap

        Args:
            explainer (Explainer): objeto explainer construído com
                        ClassifierExplainer() ou RegressionExplainer()
            title (str, optional): Título do separador ou página. Predefinição para
                        "Sumário Shap". # Nota: O default no docstring original era "Shap Dependence Summary", corrigido para "Shap Summary"
            name (str, optional): nome único a adicionar aos elementos do Componente.
                        Se None, um uuid aleatório é gerado para garantir
                        que é único. Predefinição para None.
            subtitle (str): subtítulo
            hide_title (bool, optional): ocultar o título. Predefinição para False.
            hide_subtitle (bool, optional): Ocultar subtítulo. Predefinição para False.
            hide_depth (bool, optional): ocultar o seletor de profundidade.
                        Predefinição para False.
            hide_type (bool, optional): ocultar o seletor de tipo de sumário
                        (agregado, detalhado). Predefinição para False.
            hide_popout (bool, optional): ocultar botão popout
            hide_selector (bool, optional): ocultar seletor de rótulo positivo. Predefinição para False.
            pos_label ({int, str}, optional): rótulo positivo inicial.
                        Predefinição para explainer.pos_label
            depth (int, optional): número inicial de características a mostrar. Predefinição para None.
            summary_type (str, {'aggregate', 'detailed'}. optional): tipo de
                        gráfico de sumário a mostrar. Predefinição para "aggregate".
            max_cat_colors (int, optional): para características categóricas, número máximo
                de categorias a rotular com cor própria. Predefinição para 5.
            plot_sample (int, optional): Em vez de todos os pontos, plotar apenas uma amostra
                aleatória de pontos. Predefinição para None (=todos os pontos)
            description (str, optional): Dica a exibir ao passar o rato sobre
                o título do componente. Quando None, o texto predefinido é mostrado.
        """
        super().__init__(explainer, title, name)

        if self.depth is not None:
            self.depth = min(self.depth, self.explainer.n_features)

        self.index_selector = IndexSelector(
            explainer, "shap-summary-index-" + self.name, index=index, **kwargs
        )
        self.index_name = "shap-summary-index-" + self.name
        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)
        assert self.summary_type in {"aggregate", "detailed"}
        if self.description is None:
            self.description = """
        O sumário shap resume os valores shap por característica.
        Pode selecionar uma exibição agregada que mostra o valor médio absoluto de shap
        por característica. Ou obter uma visão mais detalhada da dispersão dos valores shap por
        característica e como eles se correlacionam com o valor da característica (vermelho é alto).
        """ # Traduzido

        self.popout = GraphPopout(
            "shap-summary-" + self.name + "popout",
            "shap-summary-graph-" + self.name,
            self.title,
            self.description,
        )
        self.register_dependencies("shap_values_df")

    def layout(self):
        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        self.title, id="shap-summary-title-" + self.name # Traduzido no init
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle" # Traduzido no init
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description, # Traduzido no init
                                        target="shap-summary-title-" + self.name,
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
                                                "Profundidade:", # Traduzido
                                                id="shap-summary-depth-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                "Número de características a exibir", # Traduzido
                                                target="shap-summary-depth-label-"
                                                + self.name,
                                            ),
                                            dbc.Select(
                                                id="shap-summary-depth-" + self.name,
                                                options=[
                                                    {
                                                        "label": str(i + 1),
                                                        "value": i + 1,
                                                    }
                                                    for i in range(
                                                        self.explainer.n_features
                                                    )
                                                ],
                                                size="sm",
                                                value=self.depth,
                                            ),
                                        ],
                                        md=2,
                                    ),
                                    self.hide_depth,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Label(
                                                        "Tipo de Sumário", # Traduzido
                                                        id="shap-summary-type-label-"
                                                        + self.name,
                                                    ),
                                                    dbc.Tooltip(
                                                        "Exibir valor médio absoluto SHAP por característica (agregado)"
                                                        " ou exibir cada valor shap individual por característica (detalhado)", # Traduzido
                                                        target="shap-summary-type-label-"
                                                        + self.name,
                                                    ),
                                                    dbc.Select(
                                                        options=[
                                                            {
                                                                "label": "Agregado", # Traduzido
                                                                "value": "aggregate",
                                                            },
                                                            {
                                                                "label": "Detalhado", # Traduzido
                                                                "value": "detailed",
                                                            },
                                                        ],
                                                        value=self.summary_type,
                                                        size="sm",
                                                        id="shap-summary-type-"
                                                        + self.name,
                                                    ),
                                                ]
                                            )
                                        ]
                                    ),
                                    self.hide_type,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    dbc.Label(
                                                        f"{self.explainer.index_name}:", # Mantém variável
                                                        id="shap-summary-index-label-"
                                                        + self.name,
                                                    ),
                                                    dbc.Tooltip(
                                                        f"Selecione {self.explainer.index_name} para destacar no gráfico. "
                                                        "Também pode selecionar clicando num ponto de dispersão no gráfico.", # Traduzido
                                                        target="shap-summary-index-label-"
                                                        + self.name,
                                                    ),
                                                    self.index_selector.layout(),
                                                ],
                                                id="shap-summary-index-col-"
                                                + self.name,
                                                style=dict(display="none"), # Estilo original
                                            ),
                                        ],
                                        md=3,
                                    ),
                                    hide=self.hide_index,
                                ),
                                make_hideable(
                                    dbc.Col([self.selector.layout()], width=2),
                                    hide=self.hide_selector,
                                ),
                            ]
                        ),
                        dcc.Loading(
                            id="loading-dependence-shap-summary-" + self.name,
                            children=[
                                dcc.Graph(
                                    id="shap-summary-graph-" + self.name,
                                    config=dict(
                                        modeBarButtons=[["toImage"]], displaylogo=False
                                    ),
                                )
                            ],
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
        # Lógica original, sem tradução de eixos/colunas
        args = self.get_state_args(state_dict)
        summary_type = args.pop("summary_type")
        fig = None # Inicializa fig
        if summary_type == "aggregate":
            fig = self.explainer.plot_importances(
                kind="shap", topx=args["depth"], pos_label=args["pos_label"]
            )
        elif summary_type == "detailed":
            fig = self.explainer.plot_importances_detailed(
                topx=args["depth"],
                pos_label=args["pos_label"],
                highlight_index=args["index"],
                max_cat_colors=self.max_cat_colors,
                plot_sample=self.plot_sample,
            )

        # Verifica se fig foi gerado antes de converter para HTML
        if fig is not None:
            html_content = fig.to_html(include_plotlyjs="cdn", full_html=False)
        else:
            html_content = html.Div("Erro ao gerar o gráfico.") # Mensagem de erro genérica

        html_output = to_html.card(html_content, title=self.title, subtitle=self.subtitle)
        if add_header:
            return to_html.add_header(html_output)
        return html_output

    def component_callbacks(self, app):
        # Lógica original dos callbacks
        @app.callback(
            Output("shap-summary-index-" + self.name, "value"),
            [Input("shap-summary-graph-" + self.name, "clickData")],
            prevent_initial_call=True # Adicionado para consistência
        )
        def display_scatter_click_data(clickdata):
            if clickdata is not None and clickdata["points"]: # Verifica se points existe
                point_data = clickdata["points"][0]
                if point_data is not None and isinstance(point_data.get("y"), float):  # detailed
                    try:
                        index = point_data.get("text", "").split("=")[1].split("<br>")[0].strip()
                        # Verifica se o índice é válido
                        if self.explainer.index_exists(index):
                            return index
                    except Exception as e:
                        print(f"Erro ao extrair índice do clickData (summary): {e}")
                        pass # Ignora erros de parsing
            raise PreventUpdate

        @app.callback(
            [
                Output("shap-summary-graph-" + self.name, "figure"),
                Output("shap-summary-index-col-" + self.name, "style"),
            ],
            [
                Input("shap-summary-type-" + self.name, "value"),
                Input("shap-summary-depth-" + self.name, "value"),
                Input("shap-summary-index-" + self.name, "value"),
                Input("pos-label-" + self.name, "value"),
            ],
        )
        def update_shap_summary_graph(summary_type, depth, index, pos_label):
            depth = None if depth is None else int(depth)
            plot = go.Figure() # Figura vazia por defeito
            style_index_col = dict(display="none") # Oculto por defeito

            if summary_type == "aggregate":
                plot = self.explainer.plot_importances(
                    kind="shap", topx=depth, pos_label=pos_label
                )
                style_index_col = dict(display="none")
            elif summary_type == "detailed":
                plot = self.explainer.plot_importances_detailed(
                    topx=depth,
                    pos_label=pos_label,
                    highlight_index=index,
                    max_cat_colors=self.max_cat_colors,
                    plot_sample=self.plot_sample,
                )
                style_index_col = {} # Mostra o seletor de índice
            else:
                raise PreventUpdate # Não deve acontecer

            # Lógica original para determinar se atualiza o estilo
            ctx = dash.callback_context
            triggered_id = ""
            if ctx.triggered:
                triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

            if triggered_id == "shap-summary-type-" + self.name:
                return (plot, style_index_col)
            else:
                return (plot, dash.no_update)


class ShapDependenceComponent(ExplainerComponent):
    _state_props = dict(
        col=("shap-dependence-col-", "value"),
        color_col=("shap-dependence-color-col-", "value"),
        index=("shap-dependence-index-", "value"),
        cats_topx=("shap-dependence-n-categories-", "value"),
        cats_sort=("shap-dependence-categories-sort-", "value"),
        remove_outliers=("shap-dependence-outliers-", "value"),
        pos_label=("pos-label-", "value"),
    )

    def __init__(
        self,
        explainer,
        title="Dependência Shap", # Traduzido
        name=None,
        subtitle="Relação entre o valor da característica e o valor SHAP", # Traduzido
        hide_title=False,
        hide_subtitle=False,
        hide_col=False,
        hide_color_col=False,
        hide_index=False,
        hide_selector=False,
        hide_outliers=False,
        hide_cats_topx=False,
        hide_cats_sort=False,
        hide_popout=False,
        hide_footer=False,
        pos_label=None,
        col=None,
        color_col=None,
        index=None,
        remove_outliers=False,
        cats_topx=10,
        cats_sort="freq",
        max_cat_colors=5,
        plot_sample=None,
        description=None,
        **kwargs,
    ):
        """Mostra o gráfico de dependência shap

        Args:
            explainer (Explainer): objeto explainer construído com
                        ClassifierExplainer() ou RegressionExplainer()
            title (str, optional): Título do separador ou página. Predefinição para
                        "Dependência Shap".
            name (str, optional): nome único a adicionar aos elementos do Componente.
                        Se None, um uuid aleatório é gerado para garantir
                        que é único. Predefinição para None.
            subtitle (str): subtítulo
            hide_title (bool, optional): ocultar título do componente. Predefinição para False.
            hide_subtitle (bool, optional): Ocultar subtítulo. Predefinição para False.
            hide_col (bool, optional): ocultar seletor de característica. Predefinição para False.
            hide_color_col (bool, optional): ocultar seletor de característica de cor. Predefinição para False.
            hide_index (bool, optional): ocultar seletor de índice. Predefinição para False.
            hide_selector (bool, optional): ocultar seletor de rótulo positivo. Predefinição para False.
            hide_cats_topx (bool, optional): ocultar a entrada topx de categorias. Predefinição para False.
            hide_cats_sort (bool, optional): ocultar o seletor de ordenação de categorias. Predefinição para False.
            hide_outliers (bool, optional): Ocultar entrada de alternância para remover valores atípicos. Predefinição para False.
            hide_popout (bool, optional): ocultar botão popout. Predefinição para False.
            hide_footer (bool, optional): ocultar o rodapé.
            pos_label ({int, str}, optional): rótulo positivo inicial.
                        Predefinição para explainer.pos_label
            col (str, optional): Característica a exibir. Predefinição para None.
            color_col (str, optional): Colorir gráfico pelos valores desta Característica.
                        Predefinição para None.
            index (int, optional): Destacar um índice específico. Predefinição para None.
            remove_outliers (bool, optional): remover valores atípicos na característica e
                característica de cor do gráfico.
            cats_topx (int, optional): número máximo de categorias a exibir
                para características categóricas. Predefinição para 10.
            cats_sort (str, optional): como ordenar categorias: 'alphabet',
                'freq' ou 'shap'. Predefinição para 'freq'.
            max_cat_colors (int, optional): para características categóricas, número máximo
                de categorias a rotular com cor própria. Predefinição para 5.
            plot_sample (int, optional): Em vez de todos os pontos, plotar apenas uma amostra
                aleatória de pontos. Predefinição para None (=todos os pontos)
            description (str, optional): Dica a exibir ao passar o rato sobre
                o título do componente. Quando None, o texto predefinido é mostrado.
        """
        super().__init__(explainer, title, name)

        # Lógica original para definir col e color_col
        if self.col is None:
            self.col = self.explainer.columns_ranked_by_shap()[0]
        if self.color_col is None:
            top_interactions = self.explainer.top_shap_interactions(self.col)
            self.color_col = top_interactions[1] if len(top_interactions) > 1 else None

        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)

        self.index_selector = IndexSelector(
            explainer, "shap-dependence-index-" + self.name, index=index, **kwargs
        )
        self.index_name = "shap-dependence-index-" + self.name

        if self.description is None:
            self.description = """
        Este gráfico mostra a relação entre os valores da característica e os valores shap.
        Isto permite investigar a relação geral entre o valor da característica
        e o impacto na previsão. Pode verificar se o modelo
        usa características de acordo com as suas intuições, ou usar os gráficos para aprender
        sobre as relações que o modelo aprendeu entre as características de entrada
        e o resultado previsto.
        """ # Traduzido
        self.popout = GraphPopout(
            "shap-dependence-" + self.name + "popout",
            "shap-dependence-graph-" + self.name,
            self.title,
            self.description,
        )
        self.register_dependencies("shap_values_df")

    def layout(self):
        # Lógica original para obter opções
        col_options = [{"label": col, "value": col} for col in self.explainer.columns_ranked_by_shap()]
        initial_interact_cols = self.explainer.top_shap_interactions(self.col)
        color_col_options = [{"label": col, "value": col} for col in initial_interact_cols] + [dict(label="Nenhum", value="no_color_col")] # Traduzido "None"

        # Lógica original para verificar se a coluna é categórica
        is_col_cat = False
        try:
            # Verifica se cat_cols existe e se a coluna está nele
            if hasattr(self.explainer, 'cat_cols') and self.col in self.explainer.cat_cols:
                is_col_cat = True
        except AttributeError:
            pass # Assume False se cat_cols não existir
        style_cats_controls = {} if is_col_cat else dict(display="none")

        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        self.title, # Traduzido no init
                                        id="shap-dependence-title-" + self.name,
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle" # Traduzido no init
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description, # Traduzido no init
                                        target="shap-dependence-title-" + self.name,
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
                                    dbc.Col([self.selector.layout()], width=2),
                                    hide=self.hide_selector,
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                "Característica:", # Traduzido
                                                id="shap-dependence-col-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                "Selecione a característica para exibir a dependência shap", # Traduzido
                                                target="shap-dependence-col-label-"
                                                + self.name,
                                            ),
                                            dbc.Select(
                                                id="shap-dependence-col-" + self.name,
                                                options=col_options,
                                                value=self.col,
                                                size="sm",
                                            ),
                                        ],
                                        md=3,
                                    ),
                                    self.hide_col,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                "Característica de Cor:", # Traduzido
                                                id="shap-dependence-color-col-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                "Selecione a característica pela qual colorir os marcadores de dispersão. Isto "
                                                "permite ver interações entre várias características no gráfico.", # Traduzido
                                                target="shap-dependence-color-col-label-"
                                                + self.name,
                                            ),
                                            dbc.Select(
                                                id="shap-dependence-color-col-"
                                                + self.name,
                                                options=color_col_options,
                                                value=self.color_col if self.color_col is not None else "no_color_col",
                                                size="sm",
                                            ),
                                        ],
                                        md=3,
                                    ),
                                    self.hide_color_col,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                f"{self.explainer.index_name}:", # Mantém variável
                                                id="shap-dependence-index-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                f"Selecione {self.explainer.index_name} para destacar no gráfico."
                                                "Também pode selecionar clicando num marcador de dispersão no gráfico"
                                                " de sumário shap associado (detalhado).", # Traduzido
                                                target="shap-dependence-index-label-"
                                                + self.name,
                                            ),
                                            self.index_selector.layout(),
                                        ],
                                        md=4,
                                    ),
                                    hide=self.hide_index,
                                ),
                            ]
                        ),
                        dcc.Loading(
                            id="loading-dependence-graph-" + self.name,
                            children=[
                                dcc.Graph(
                                    id="shap-dependence-graph-" + self.name,
                                    config=dict(
                                        modeBarButtons=[["toImage"]], displaylogo=False
                                    ),
                                )
                            ],
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
                                                        dbc.Tooltip(
                                                            "Remover valores atípicos na característica (e característica de cor) do gráfico.", # Traduzido
                                                            target="shap-dependence-outliers-"
                                                            + self.name,
                                                        ),
                                                        dbc.Checklist(
                                                            options=[
                                                                {
                                                                    "label": "Remover valores atípicos", # Traduzido
                                                                    "value": True,
                                                                }
                                                            ],
                                                            value=[True]
                                                            if self.remove_outliers
                                                            else [],
                                                            id="shap-dependence-outliers-"
                                                            + self.name,
                                                            inline=True,
                                                            switch=True,
                                                        ),
                                                    ]
                                                ),
                                            ],
                                            md=2,
                                        ),
                                        hide=self.hide_outliers,
                                    ),
                                    make_hideable(
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    [
                                                        dbc.Label(
                                                            "Categorias:", # Traduzido
                                                            id="shap-dependence-n-categories-label-"
                                                            + self.name,
                                                        ),
                                                        dbc.Tooltip(
                                                            "Número máximo de categorias a exibir", # Traduzido
                                                            target="shap-dependence-n-categories-label-"
                                                            + self.name,
                                                        ),
                                                        dbc.Input(
                                                            id="shap-dependence-n-categories-"
                                                            + self.name,
                                                            value=self.cats_topx,
                                                            type="number",
                                                            min=1,
                                                            max=50,
                                                            step=1,
                                                            size="sm", # Adicionado size
                                                        ),
                                                    ],
                                                    id="shap-dependence-categories-div1-"
                                                    + self.name,
                                                    style=style_cats_controls, # Estilo original
                                                )
                                            ],
                                            md=2,
                                        ),
                                        self.hide_cats_topx,
                                    ),
                                    make_hideable(
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    [
                                                        html.Label(
                                                            "Ordenar categorias:", # Traduzido
                                                            id="shap-dependence-categories-sort-label-"
                                                            + self.name,
                                                        ),
                                                        dbc.Tooltip(
                                                            "Como ordenar as categorias: Alfabeticamente, mais comuns "
                                                            "primeiro (Frequência), ou maior valor médio absoluto SHAP primeiro (Impacto Shap)", # Traduzido
                                                            target="shap-dependence-categories-sort-label-"
                                                            + self.name,
                                                        ),
                                                        dbc.Select(
                                                            id="shap-dependence-categories-sort-"
                                                            + self.name,
                                                            options=[
                                                                {
                                                                    "label": "Alfabeticamente", # Traduzido
                                                                    "value": "alphabet",
                                                                },
                                                                {
                                                                    "label": "Frequência", # Traduzido
                                                                    "value": "freq",
                                                                },
                                                                {
                                                                    "label": "Impacto Shap", # Traduzido
                                                                    "value": "shap",
                                                                },
                                                            ],
                                                            value=self.cats_sort,
                                                            size="sm",
                                                        ),
                                                    ],
                                                    id="shap-dependence-categories-div2-"
                                                    + self.name,
                                                    style=style_cats_controls, # Estilo original
                                                )
                                            ],
                                            md=4,
                                        ),
                                        hide=self.hide_cats_sort,
                                    ),
                                ]
                            )
                        ]
                    ),
                    hide=self.hide_footer,
                ),
            ],
            class_name="h-100",
        )

    def to_html(self, state_dict=None, add_header=True):
        # Lógica original, sem tradução de eixos/colunas
        args = self.get_state_args(state_dict)

        _color_col = args["color_col"]
        _index = args["index"]
        if _color_col == "no_color_col":
            _color_col, _index = None, None

        fig = self.explainer.plot_dependence(
            args["col"],
            _color_col,
            topx=args["cats_topx"],
            sort=args["cats_sort"],
            highlight_index=_index,
            max_cat_colors=self.max_cat_colors,
            plot_sample=self.plot_sample,
            remove_outliers=bool(args["remove_outliers"]),
            pos_label=args["pos_label"],
        )

        html_content = fig.to_html(include_plotlyjs="cdn", full_html=False)
        html_output = to_html.card(html_content, title=self.title, subtitle=self.subtitle)
        if add_header:
            return to_html.add_header(html_output)
        return html_output

    def component_callbacks(self, app):
        # Lógica original dos callbacks
        @app.callback(
            [
                Output("shap-dependence-color-col-" + self.name, "options"),
                Output("shap-dependence-color-col-" + self.name, "value"),
                Output("shap-dependence-categories-div1-" + self.name, "style"),
                Output("shap-dependence-categories-div2-" + self.name, "style"),
            ],
            [Input("shap-dependence-col-" + self.name, "value")],
            [State("pos-label-" + self.name, "value"),
             State("shap-dependence-color-col-" + self.name, "value")], # Adicionado estado
        )
        def set_color_col_dropdown(col, pos_label, current_color_col): # Adicionado current_color_col
            if col is None:
                raise PreventUpdate

            sorted_interact_cols = self.explainer.top_shap_interactions(
                col, pos_label=pos_label
            )
            options = [{"label": c, "value": c} for c in sorted_interact_cols] + [
                dict(label="Nenhum", value="no_color_col") # Traduzido "None"
            ]

            # Lógica original para determinar valor da cor
            new_color_col_val = "no_color_col" # Predefinição
            if len(sorted_interact_cols) > 1:
                 if current_color_col in sorted_interact_cols and current_color_col != col:
                     new_color_col_val = current_color_col
                 elif sorted_interact_cols[1] != col:
                     new_color_col_val = sorted_interact_cols[1]
                 elif len(sorted_interact_cols) > 2 and sorted_interact_cols[2] != col:
                     new_color_col_val = sorted_interact_cols[2]

            # Lógica original para determinar estilo
            style = dict(display="none")
            try:
                if hasattr(self.explainer, 'cat_cols') and col in self.explainer.cat_cols:
                    style = {}
            except AttributeError: pass

            return (options, new_color_col_val, style, style)

        @app.callback(
            Output("shap-dependence-graph-" + self.name, "figure"),
            [
                Input("shap-dependence-color-col-" + self.name, "value"),
                Input("shap-dependence-index-" + self.name, "value"),
                Input("shap-dependence-n-categories-" + self.name, "value"),
                Input("shap-dependence-categories-sort-" + self.name, "value"),
                Input("shap-dependence-outliers-" + self.name, "value"),
                Input("pos-label-" + self.name, "value"),
            ],
            [State("shap-dependence-col-" + self.name, "value")],
        )
        def update_dependence_graph(
            color_col, index, topx, sort, remove_outliers, pos_label, col
        ):
            if col is not None:
                _color_col = color_col
                _index = index
                if color_col == "no_color_col":
                    _color_col, _index = None, None

                fig = self.explainer.plot_dependence(
                    col,
                    _color_col,
                    topx=topx,
                    sort=sort,
                    highlight_index=_index,
                    max_cat_colors=self.max_cat_colors,
                    plot_sample=self.plot_sample,
                    remove_outliers=bool(remove_outliers),
                    pos_label=pos_label,
                )
                return fig
            return go.Figure() # Retorna figura vazia se col for None


class ShapSummaryDependenceConnector(ExplainerComponent):
    # Sem interface de utilizador, sem tradução necessária.
    def __init__(self, shap_summary_component, shap_dependence_component):
        """Connects a ShapSummaryComponent with a ShapDependence Component:

        - When clicking on feature in ShapSummary, then select that feature in ShapDependence

        Args:
            shap_summary_component (ShapSummaryComponent): ShapSummaryComponent
            shap_dependence_component (ShapDependenceComponent): ShapDependenceComponent
        """
        self.sum_name = shap_summary_component.name
        self.dep_name = shap_dependence_component.name

    def component_callbacks(self, app):
        # Lógica original do callback
        @app.callback(
            [
                Output("shap-dependence-index-" + self.dep_name, "value"),
                Output("shap-dependence-col-" + self.dep_name, "value"),
            ],
            [Input("shap-summary-graph-" + self.sum_name, "clickData")],
            prevent_initial_call=True # Mantido
        )
        def display_scatter_click_data(clickdata):
            if clickdata is not None and clickdata["points"]: # Verifica points
                point_data = clickdata["points"][0]
                if point_data is not None:
                    if isinstance(point_data.get("y"), float):  # detailed
                        try:
                            text_parts = point_data.get("text", "").split("<br>")
                            if len(text_parts) >= 2:
                                index_part = text_parts[0].split("=")
                                col_part = text_parts[1].split("=") # Assume Feature=col_name
                                if len(index_part) > 1 and len(col_part) > 1:
                                    index = index_part[1].strip()
                                    col = col_part[1].strip()
                                    if self.explainer.index_exists(index) and col in self.explainer.columns:
                                         return (index, col)
                        except Exception as e:
                             print(f"Erro ao processar clickData (connector detailed): {e}")
                             pass
                    elif isinstance(point_data.get("y"), str):  # aggregate
                        try:
                            col = point_data["y"].split(":")[-1].strip() # Tenta obter após ':'
                            if col in self.explainer.columns:
                                return (dash.no_update, col)
                        except Exception as e:
                            print(f"Erro ao processar clickData (connector aggregate): {e}")
                            pass
            raise PreventUpdate


class InteractionSummaryComponent(ExplainerComponent):
    _state_props = dict(
        col=("interaction-summary-col-", "value"),
        depth=("interaction-summary-depth-", "value"),
        summary_type=("interaction-summary-type-", "value"),
        index=("interaction-summary-index-", "value"),
        pos_label=("pos-label-", "value"),
    )

    def __init__(
        self,
        explainer,
        title="Sumário de Interações", # Traduzido
        name=None,
        subtitle="Ordenando características por valor de interação shap", # Traduzido
        hide_title=False,
        hide_subtitle=False,
        hide_col=False,
        hide_depth=False,
        hide_type=False,
        hide_index=False,
        hide_popout=False,
        hide_selector=False,
        pos_label=None,
        col=None,
        depth=None,
        summary_type="aggregate",
        max_cat_colors=5,
        index=None,
        plot_sample=None,
        description=None,
        **kwargs,
    ):
        """Componente de sumário de valores de Interação SHAP

        Args:
            explainer (Explainer): objeto explainer construído com
                        ClassifierExplainer() ou RegressionExplainer()
            title (str, optional): Título do separador ou página. Predefinição para
                        "Sumário de Interações".
            name (str, optional): nome único a adicionar aos elementos do Componente.
                        Se None, um uuid aleatório é gerado para garantir
                        que é único. Predefinição para None.
            subtitle (str): subtítulo
            hide_title (bool, optional): ocultar o título do componente. Predefinição para False.
            hide_subtitle (bool, optional): Ocultar subtítulo. Predefinição para False.
            hide_col (bool, optional): Ocultar o seletor de característica. Predefinição para False.
            hide_depth (bool, optional): Ocultar seletor de profundidade. Predefinição para False.
            hide_type (bool, optional): Ocultar seletor de tipo de sumário. Predefinição para False.
            hide_index (bool, optional): Ocultar o seletor de índice. Predefinição para False
            hide_popout (bool, optional): ocultar botão popout
            hide_selector (bool, optional): ocultar seletor de rótulo positivo. Predefinição para False.
            pos_label ({int, str}, optional): rótulo positivo inicial.
                        Predefinição para explainer.pos_label
            col (str, optional): Característica para mostrar sumário de interação.
                Predefinição para None.
            depth (int, optional): Número de características de interação a exibir.
                Predefinição para None.
            summary_type (str, {'aggregate', 'detailed'}, optional): tipo de
                gráfico de sumário a exibir. Predefinição para "aggregate".
            max_cat_colors (int, optional): para características categóricas, número máximo
                de categorias a rotular com cor própria. Predefinição para 5.
            index (str):    Índice predefinido. Predefinição para None.
            plot_sample (int, optional): Em vez de todos os pontos, plotar apenas uma amostra
                aleatória de pontos. Predefinição para None (=todos os pontos)
            description (str, optional): Dica a exibir ao passar o rato sobre
                o título do componente. Quando None, o texto predefinido é mostrado.
        """
        super().__init__(explainer, title, name)

        # Lógica original para definir col e depth
        if self.col is None:
            self.col = self.explainer.columns_ranked_by_shap()[0]
        if self.depth is not None:
            self.depth = min(self.depth, self.explainer.n_features - 1 if self.explainer.n_features > 1 else 1)

        self.index_selector = IndexSelector(
            explainer, "interaction-summary-index-" + self.name, index=index, **kwargs
        )
        self.index_name = "interaction-summary-index-" + self.name
        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)

        if self.description is None:
            self.description = """
        Mostra os valores de interação shap. Cada valor shap pode ser decomposto num
        efeito direto e efeitos indiretos. Os efeitos indiretos devem-se a interações
        da característica com outras características. Por exemplo, saber
        o género de um passageiro no Titanic terá um efeito direto (mulheres
        com maior probabilidade de sobreviver do que homens), mas também pode ter efeitos indiretos através,
        por exemplo, da classe do passageiro (mulheres de primeira classe com maior probabilidade de sobreviver do que
        a mulher média, mulheres de terceira classe com menor probabilidade).
        """ # Traduzido
        self.popout = GraphPopout(
            "interaction-summary-" + self.name + "popout",
            "interaction-summary-graph-" + self.name,
            self.title,
            self.description,
        )
        # Lógica original de aviso e dependência
        if not hasattr(explainer, 'shap_interaction_values') or explainer.shap_interaction_values is None:
             print("Warning: InteractionSummaryComponent requires explainer.shap_interaction_values. " \
                   "Please calculate interaction values first, e.g. explainer.calculate_properties(include_interactions=True)")
        self.register_dependencies("shap_interaction_values")

    def layout(self):
        # Lógica original para obter opções
        col_options = [{"label": col, "value": col} for col in self.explainer.columns_ranked_by_shap()]
        depth_options = [{"label": str(i + 1), "value": i + 1} for i in range(self.explainer.n_features - 1 if self.explainer.n_features > 1 else 0)]

        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        self.title, # Traduzido no init
                                        id="interaction-summary-title-" + self.name,
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle" # Traduzido no init
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description, # Traduzido no init
                                        target="interaction-summary-title-" + self.name,
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
                                                "Característica", # Traduzido
                                                id="interaction-summary-col-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                "Característica para selecionar efeitos de interações", # Traduzido
                                                target="interaction-summary-col-label-"
                                                + self.name,
                                            ),
                                            dbc.Select(
                                                id="interaction-summary-col-"
                                                + self.name,
                                                options=col_options,
                                                value=self.col,
                                                size="sm", # Adicionado size
                                            ),
                                        ],
                                        md=2,
                                    ),
                                    self.hide_col,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                "Profundidade:", # Traduzido
                                                id="interaction-summary-depth-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                "Número de características de interação a exibir", # Traduzido
                                                target="interaction-summary-depth-label-"
                                                + self.name,
                                            ),
                                            dbc.Select(
                                                id="interaction-summary-depth-"
                                                + self.name,
                                                options=depth_options,
                                                value=self.depth,
                                                size="sm", # Adicionado size
                                            ),
                                        ],
                                        md=2,
                                    ),
                                    self.hide_depth,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Label(
                                                        "Tipo de Sumário", # Traduzido
                                                        id="interaction-summary-type-label-"
                                                        + self.name,
                                                    ),
                                                    dbc.Tooltip(
                                                        "Exibir valor médio absoluto SHAP por característica (agregado)"
                                                        " ou exibir cada valor shap individual por característica (detalhado)", # Traduzido
                                                        target="interaction-summary-type-label-"
                                                        + self.name,
                                                    ),
                                                    dbc.Select(
                                                        options=[
                                                            {
                                                                "label": "Agregado", # Traduzido
                                                                "value": "aggregate",
                                                            },
                                                            {
                                                                "label": "Detalhado", # Traduzido
                                                                "value": "detailed",
                                                            },
                                                        ],
                                                        value=self.summary_type,
                                                        id="interaction-summary-type-"
                                                        + self.name,
                                                        size="sm", # Adicionado size
                                                    ),
                                                ]
                                            )
                                        ],
                                        md=3,
                                    ),
                                    self.hide_type,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    dbc.Label(
                                                        f"{self.explainer.index_name}:", # Mantém variável
                                                        id="interaction-summary-index-label-"
                                                        + self.name,
                                                    ),
                                                    dbc.Tooltip(
                                                        f"Selecione {self.explainer.index_name} para destacar no gráfico. "
                                                        "Também pode selecionar clicando num ponto de dispersão no gráfico.", # Traduzido
                                                        target="interaction-summary-index-label-"
                                                        + self.name,
                                                    ),
                                                    self.index_selector.layout(),
                                                ],
                                                id="interaction-summary-index-col-"
                                                + self.name,
                                                style=dict(display="none"), # Estilo original
                                            ),
                                        ],
                                        md=3,
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
                                            id="loading-interaction-summary-graph-"
                                            + self.name,
                                            children=[
                                                dcc.Graph(
                                                    id="interaction-summary-graph-"
                                                    + self.name,
                                                    config=dict(
                                                        modeBarButtons=[["toImage"]],
                                                        displaylogo=False,
                                                    ),
                                                )
                                            ],
                                        )
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
            ],
            class_name="h-100",
        )

    def to_html(self, state_dict=None, add_header=True):
        # Lógica original, sem tradução de eixos/colunas
        args = self.get_state_args(state_dict)
        html_content = "" # Inicializa
        if not hasattr(self.explainer, 'shap_interaction_values') or self.explainer.shap_interaction_values is None:
             html_content = html.Div("Valores de interação SHAP não calculados. Por favor, calcule-os primeiro.") # Traduzido
        elif args["col"] is None:
             html_content = html.Div("Por favor, selecione uma característica.") # Traduzido
        elif args["summary_type"] == "aggregate":
            fig = self.explainer.plot_interactions_importance(
                args["col"], topx=args["depth"], pos_label=args["pos_label"]
            )
            html_content = to_html.fig(fig)
        elif args["summary_type"] == "detailed":
            fig = self.explainer.plot_interactions_detailed(
                args["col"],
                topx=args["depth"],
                pos_label=args["pos_label"],
                highlight_index=args["index"],
                max_cat_colors=self.max_cat_colors,
                plot_sample=self.plot_sample,
            )
            html_content = to_html.fig(fig)
        else:
            html_content = html.Div("Tipo de sumário inválido.") # Traduzido

        html_output = to_html.card(html_content, title=self.title, subtitle=self.subtitle)
        if add_header:
            return to_html.add_header(html_output)
        return html_output

    def component_callbacks(self, app):
        # Lógica original dos callbacks
        @app.callback(
            Output("interaction-summary-index-" + self.name, "value"),
            [Input("interaction-summary-graph-" + self.name, "clickData")],
            prevent_initial_call=True # Mantido
        )
        def display_scatter_click_data(clickdata):
            if clickdata is not None and clickdata["points"]: # Verifica points
                point_data = clickdata["points"][0]
                if point_data is not None and isinstance(point_data.get("y"), float):  # detailed graph
                    try:
                        text_parts = point_data.get("text", "").split("<br>")
                        if len(text_parts) >= 1:
                            index_part = text_parts[0].split("=")
                            if len(index_part) > 1:
                                index = index_part[1].strip()
                                if self.explainer.index_exists(index):
                                    return index
                    except Exception as e:
                        print(f"Erro ao extrair índice do clickData (interaction summary): {e}")
                        pass
            raise PreventUpdate

        @app.callback(
            [
                Output("interaction-summary-graph-" + self.name, "figure"),
                Output("interaction-summary-index-col-" + self.name, "style"),
            ],
            [
                Input("interaction-summary-col-" + self.name, "value"),
                Input("interaction-summary-depth-" + self.name, "value"),
                Input("interaction-summary-type-" + self.name, "value"),
                Input("interaction-summary-index-" + self.name, "value"),
                Input("pos-label-" + self.name, "value"),
            ],
        )
        def update_interaction_scatter_graph(
            col, depth, summary_type, index, pos_label
        ):
            if not hasattr(self.explainer, 'shap_interaction_values') or self.explainer.shap_interaction_values is None:
                 return go.Figure(), dict(display="none")

            plot = go.Figure() # Figura vazia por defeito
            style_index_col = dict(display="none") # Oculto por defeito

            if col is not None:
                depth = None if depth is None else int(depth)
                if summary_type == "aggregate":
                    plot = self.explainer.plot_interactions_importance(
                        col, topx=depth, pos_label=pos_label
                    )
                    style_index_col = dict(display="none")
                elif summary_type == "detailed":
                    plot = self.explainer.plot_interactions_detailed(
                        col,
                        topx=depth,
                        pos_label=pos_label,
                        highlight_index=index,
                        max_cat_colors=self.max_cat_colors,
                        plot_sample=self.plot_sample,
                    )
                    style_index_col = {} # Mostra o seletor de índice
                else:
                    raise PreventUpdate # Não deve acontecer

            # Lógica original para determinar se atualiza o estilo
            ctx = dash.callback_context
            triggered_id = ""
            if ctx.triggered:
                triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

            if triggered_id == "interaction-summary-type-" + self.name:
                return (plot, style_index_col)
            else:
                return (plot, dash.no_update)


class InteractionDependenceComponent(ExplainerComponent):
    # Mantendo os _state_props originais
    _state_props = dict(
        col=("interaction-dependence-col-", "value"),
        interact_col=("interaction-dependence-interact-col-", "value"),
        index=("interaction-dependence-index-", "value"),
        cats_topx=("interaction-dependence-top-n-categories-", "value"),
        cats_sort=("interaction-dependence-top-categories-sort-", "value"),
        remove_outliers=("interaction-dependence-top-outliers-", "value"),
        # Adiciona props para o gráfico inferior (mantendo nomes originais)
        cats_topx_bottom=("interaction-dependence-bottom-n-categories-", "value"),
        cats_sort_bottom=("interaction-dependence-bottom-categories-sort-", "value"),
        remove_outliers_bottom=("interaction-dependence-bottom-outliers-", "value"),
        pos_label=("pos-label-", "value"),
    )

    def __init__(
        self,
        explainer,
        title="Dependência de Interação", # Traduzido
        name=None,
        subtitle="Relação entre o valor da característica e o valor de interação shap", # Traduzido
        hide_title=False,
        hide_subtitle=False,
        hide_col=False,
        hide_interact_col=False,
        hide_index=False,
        hide_popout=False,
        hide_selector=False,
        hide_outliers=False,
        hide_cats_topx=False,
        hide_cats_sort=False,
        hide_top=False,
        hide_bottom=False,
        pos_label=None,
        col=None,
        interact_col=None,
        remove_outliers=False,
        cats_topx=10,
        cats_sort="freq",
        # Mantendo os parâmetros originais para _bottom
        remove_outliers_bottom=None,
        cats_topx_bottom=None,
        cats_sort_bottom=None,
        max_cat_colors=5,
        plot_sample=None,
        description=None,
        index=None,
        **kwargs,
    ):
        """Componente de Dependência de Interação.

        Mostra dois gráficos:
            gráfico superior: col vs interact_col
            gráfico inferior: interact_col vs col

        Args:
            explainer (Explainer): objeto explainer construído com
                        ClassifierExplainer() ou RegressionExplainer()
            title (str, optional): Título do separador ou página. Predefinição para
                        "Dependência de Interação".
            name (str, optional): nome único a adicionar aos elementos do Componente.
                        Se None, um uuid aleatório é gerado para garantir
                        que é único. Predefinição para None.
            subtitle (str): subtítulo
            hide_title (bool, optional): Ocultar título do componente. Predefinição para False.
            hide_subtitle (bool, optional): Ocultar subtítulo. Predefinição para False.
            hide_col (bool, optional): Ocultar seletor de característica. Predefinição para False.
            hide_interact_col (bool, optional): Ocultar seletor de característica
                        de interação. Predefinição para False.
            hide_index (bool, optional): Ocultar seletor de índice de destaque.
                        Predefinição para False. # Nota: hide_highlight não existia, mantido hide_index
            hide_selector (bool, optional): ocultar seletor de rótulo positivo.
                        Predefinição para False.
            hide_outliers (bool, optional): Ocultar entrada de alternância para remover valores atípicos (para ambos os gráficos). Predefinição para False.
            hide_popout (bool, optional): ocultar botão popout (para ambos os gráficos).
            hide_cats_topx (bool, optional): ocultar a entrada topx de categorias (para ambos os gráficos).
                        Predefinição para False.
            hide_cats_sort (bool, optional): ocultar o seletor de ordenação de categorias (para ambos os gráficos).
                        Predefinição para False.
            hide_top (bool, optional): Ocultar o gráfico de interação superior
                        (col vs interact_col). Predefinição para False.
            hide_bottom (bool, optional): ocultar o gráfico de interação inferior
                        (interact_col vs col). Predefinição para False.
            pos_label ({int, str}, optional): rótulo positivo inicial.
                        Predefinição para explainer.pos_label
            col (str, optional): Característica para encontrar interações. Predefinição para None.
            interact_col (str, optional): Característica com a qual interagir. Predefinição para None.
            index (int, optional): Linha de índice a destacar Defaults to None. # Mantido index em vez de highlight
            remove_outliers (bool, optional): remover valores atípicos na característica e
                característica de cor do gráfico superior.
            cats_topx (int, optional): número de categorias a exibir para
                características categóricas no gráfico superior.
            cats_sort (str, optional): como ordenar categorias no gráfico superior: 'alphabet',
                'freq' ou 'shap'. Predefinição para 'freq'.
            remove_outliers_bottom (bool, optional): remover valores atípicos no gráfico inferior. Predefinição para `remove_outliers`.
            cats_topx_bottom (int, optional): número de categorias a exibir no gráfico inferior. Predefinição para `cats_topx`.
            cats_sort_bottom (str, optional): como ordenar categorias no gráfico inferior. Predefinição para `cats_sort`.
            max_cat_colors (int, optional): para características categóricas, número máximo
                de categorias a rotular com cor própria. Predefinição para 5.
            plot_sample (int, optional): Em vez de todos os pontos, plotar apenas uma amostra
                aleatória de pontos. Predefinição para None (=todos os pontos)
            description (str, optional): Dica a exibir ao passar o rato sobre
                o título do componente. Quando None, o texto predefinido é mostrado.
        """
        super().__init__(explainer, title, name)

        # Lógica original para definir col e interact_col
        if self.col is None:
            self.col = explainer.columns_ranked_by_shap()[0]
        if self.interact_col is None:
            top_interactions = explainer.top_shap_interactions(self.col)
            self.interact_col = next((c for c in top_interactions if c != self.col), None)

        # Lógica original para definir valores _bottom
        self.remove_outliers_bottom = remove_outliers if remove_outliers_bottom is None else remove_outliers_bottom
        self.cats_topx_bottom = cats_topx if cats_topx_bottom is None else cats_topx_bottom
        self.cats_sort_bottom = cats_sort if cats_sort_bottom is None else cats_sort_bottom

        self.index_selector = IndexSelector(
            explainer,
            "interaction-dependence-index-" + self.name,
            index=index,
            **kwargs,
        )
        self.index_name = "interaction-dependence-index-" + self.name

        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)
        self.popout_top = GraphPopout(
            self.name + "popout-top",
            "interaction-dependence-top-graph-" + self.name,
            self.title + " (Superior)", # Traduzido
        )

        if self.description is None:
            self.description = """
        Este gráfico mostra a relação entre os valores da característica e os valores de interação shap.
        Isto permite investigar interações entre características na determinação
        da previsão do modelo. São mostrados dois gráficos: a interação de A com B e a interação de B com A.
        """ # Traduzido
        self.popout_bottom = GraphPopout(
            self.name + "popout-bottom",
            "interaction-dependence-bottom-graph-" + self.name,
            self.title + " (Inferior)", # Traduzido
            self.description,
        )
        # Lógica original de aviso e dependência
        if not hasattr(explainer, 'shap_interaction_values') or explainer.shap_interaction_values is None:
             print("Warning: InteractionDependenceComponent requires explainer.shap_interaction_values. " \
                   "Please calculate interaction values first, e.g. explainer.calculate_properties(include_interactions=True)")
        self.register_dependencies("shap_interaction_values")

    def layout(self):
        # Lógica original para obter opções
        col_options = [{"label": col, "value": col} for col in self.explainer.columns_ranked_by_shap()]
        initial_interact_cols = self.explainer.top_shap_interactions(self.col)
        interact_col_options = [{"label": c, "value": c} for c in initial_interact_cols if c != self.col]

        # Lógica original para definir valor inicial de interact_col
        current_interact_col = self.interact_col
        if current_interact_col not in [opt['value'] for opt in interact_col_options]:
             current_interact_col = interact_col_options[0]['value'] if interact_col_options else None

        # Lógica original para estilos condicionais
        is_interact_col_cat = False
        is_col_cat = False
        try:
            if hasattr(self.explainer, 'cat_cols'):
                if current_interact_col in self.explainer.cat_cols:
                    is_interact_col_cat = True
                if self.col in self.explainer.cat_cols:
                    is_col_cat = True
        except AttributeError: pass
        style_top_cats = {} if is_interact_col_cat else dict(display="none")
        style_bottom_cats = {} if is_col_cat else dict(display="none")

        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        self.title, # Traduzido no init
                                        id="interaction-dependence-title-" + self.name,
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle" # Traduzido no init
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description, # Traduzido no init
                                        target="interaction-dependence-title-"
                                        + self.name,
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
                                    dbc.Col([self.selector.layout()], width=2),
                                    hide=self.hide_selector,
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                "Característica:", # Traduzido
                                                id="interaction-dependence-col-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                "Selecione a característica principal para exibir interações shap", # Traduzido
                                                target="interaction-dependence-col-label-"
                                                + self.name,
                                            ),
                                            dbc.Select(
                                                id="interaction-dependence-col-"
                                                + self.name,
                                                options=col_options,
                                                value=self.col,
                                                size="sm", # Adicionado size
                                            ),
                                        ],
                                        md=3,
                                    ),
                                    hide=self.hide_col,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            html.Label(
                                                "Interação:", # Traduzido
                                                id="interaction-dependence-interact-col-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                "Selecione a característica de interação. Serão mostrados dois gráficos: "
                                                "Característica vs Interação e Interação vs Característica.", # Traduzido
                                                target="interaction-dependence-interact-col-label-"
                                                + self.name,
                                            ),
                                            dbc.Select(
                                                id="interaction-dependence-interact-col-"
                                                + self.name,
                                                options=interact_col_options,
                                                value=current_interact_col,
                                                size="sm", # Adicionado size
                                            ),
                                        ],
                                        md=3,
                                    ),
                                    hide=self.hide_interact_col,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                f"{self.explainer.index_name}:", # Mantém variável
                                                id="interaction-dependence-index-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                f"Selecione {self.explainer.index_name} para destacar nos gráficos."
                                                "Também pode selecionar clicando num marcador de dispersão no gráfico"
                                                " de sumário de interação associado (detalhado).", # Traduzido
                                                target="interaction-dependence-index-label-"
                                                + self.name,
                                            ),
                                            self.index_selector.layout(),
                                        ],
                                        md=4,
                                    ),
                                    hide=self.hide_index,
                                ),
                            ]
                        ),
                        # Gráfico Superior
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        make_hideable(
                                            dcc.Loading(
                                                id="loading-interaction-dependence-top-graph-"
                                                + self.name,
                                                children=[
                                                    dcc.Graph(
                                                        id="interaction-dependence-top-graph-"
                                                        + self.name,
                                                        config=dict(
                                                            modeBarButtons=[
                                                                ["toImage"]
                                                            ],
                                                            displaylogo=False,
                                                        ),
                                                    )
                                                ],
                                            ),
                                            hide=self.hide_top,
                                        ),
                                    ]
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [self.popout_top.layout()], md=2, align="start"
                                    ),
                                    hide=self.hide_popout,
                                ),
                            ],
                            justify="end",
                        ),
                        # Controlos do Gráfico Superior
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Tooltip(
                                                        "Remover valores atípicos (> 1.5*IQR) na característica e característica de interação do gráfico superior.", # Traduzido
                                                        target="interaction-dependence-top-outliers-"
                                                        + self.name,
                                                    ),
                                                    dbc.Checklist(
                                                        options=[
                                                            {
                                                                "label": "Remover valores atípicos (Superior)", # Traduzido
                                                                "value": True,
                                                            }
                                                        ],
                                                        value=[True]
                                                        if self.remove_outliers
                                                        else [],
                                                        id="interaction-dependence-top-outliers-"
                                                        + self.name,
                                                        inline=True,
                                                        switch=True,
                                                    ),
                                                ]
                                            ),
                                        ],
                                        md=3,
                                    ),
                                    hide=self.hide_outliers,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    dbc.Label(
                                                        "Categorias (Sup):", # Traduzido
                                                        id="interaction-dependence-top-n-categories-label-"
                                                        + self.name,
                                                    ),
                                                    dbc.Tooltip(
                                                        "Número máximo de categorias a exibir (gráfico superior)", # Traduzido
                                                        target="interaction-dependence-top-n-categories-label-"
                                                        + self.name,
                                                    ),
                                                    dbc.Input(
                                                        id="interaction-dependence-top-n-categories-"
                                                        + self.name,
                                                        value=self.cats_topx,
                                                        type="number",
                                                        min=1, max=50, step=1, size="sm"
                                                    ),
                                                ],
                                                id="interaction-dependence-top-categories-div1-"
                                                + self.name,
                                                style=style_top_cats, # Estilo original
                                            ),
                                        ],
                                        md=3,
                                    ),
                                    self.hide_cats_topx,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Ordenar Cat. (Sup):", # Traduzido
                                                        id="interaction-dependence-top-categories-sort-label-"
                                                        + self.name,
                                                    ),
                                                    dbc.Tooltip(
                                                        "Como ordenar as categorias (gráfico superior): Alfabeticamente, Frequência, ou Impacto Shap", # Traduzido
                                                        target="interaction-dependence-top-categories-sort-label-"
                                                        + self.name,
                                                    ),
                                                    dbc.Select(
                                                        id="interaction-dependence-top-categories-sort-"
                                                        + self.name,
                                                        options=[
                                                            {"label": "Alfabeticamente", "value": "alphabet"},
                                                            {"label": "Frequência", "value": "freq"},
                                                            {"label": "Impacto Shap", "value": "shap"},
                                                        ],
                                                        value=self.cats_sort,
                                                        size="sm",
                                                    ),
                                                ],
                                                id="interaction-dependence-top-categories-div2-"
                                                + self.name,
                                                style=style_top_cats, # Estilo original
                                            ),
                                        ],
                                        md=4,
                                    ),
                                    hide=self.hide_cats_sort,
                                ),
                            ]
                        ),
                        html.Hr(), # Separador original
                        # Gráfico Inferior
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        make_hideable(
                                            dcc.Loading(
                                                id="loading-reverse-interaction-bottom-graph-"
                                                + self.name,
                                                children=[
                                                    dcc.Graph(
                                                        id="interaction-dependence-bottom-graph-"
                                                        + self.name,
                                                        config=dict(
                                                            modeBarButtons=[
                                                                ["toImage"]
                                                            ],
                                                            displaylogo=False,
                                                        ),
                                                    )
                                                ],
                                            ),
                                            hide=self.hide_bottom,
                                        ),
                                    ]
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [self.popout_bottom.layout()],
                                        md=2,
                                        align="start",
                                    ),
                                    hide=self.hide_popout,
                                ),
                            ],
                            justify="end",
                        ),
                        # Controlos do Gráfico Inferior
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Tooltip(
                                                        "Remover valores atípicos (> 1.5*IQR) na característica e característica de interação do gráfico inferior.", # Traduzido
                                                        target="interaction-dependence-bottom-outliers-"
                                                        + self.name,
                                                    ),
                                                    dbc.Checklist(
                                                        options=[
                                                            {
                                                                "label": "Remover valores atípicos (Inferior)", # Traduzido
                                                                "value": True,
                                                            }
                                                        ],
                                                        value=[True]
                                                        if self.remove_outliers_bottom
                                                        else [],
                                                        id="interaction-dependence-bottom-outliers-"
                                                        + self.name,
                                                        inline=True,
                                                        switch=True,
                                                    ),
                                                ]
                                            ),
                                        ],
                                        md=3,
                                    ),
                                    hide=self.hide_outliers,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    dbc.Label(
                                                        "Categorias (Inf):", # Traduzido
                                                        id="interaction-dependence-bottom-n-categories-label-"
                                                        + self.name,
                                                    ),
                                                    dbc.Tooltip(
                                                        "Número máximo de categorias a exibir (gráfico inferior)", # Traduzido
                                                        target="interaction-dependence-bottom-n-categories-label-"
                                                        + self.name,
                                                    ),
                                                    dbc.Input(
                                                        id="interaction-dependence-bottom-n-categories-"
                                                        + self.name,
                                                        value=self.cats_topx_bottom,
                                                        type="number",
                                                        min=1, max=50, step=1, size="sm"
                                                    ),
                                                ],
                                                id="interaction-dependence-bottom-categories-div1-"
                                                + self.name,
                                                style=style_bottom_cats, # Estilo original
                                            ),
                                        ],
                                        md=3,
                                    ),
                                    self.hide_cats_topx,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Ordenar Cat. (Inf):", # Traduzido
                                                        id="interaction-dependence-bottom-categories-sort-label-"
                                                        + self.name,
                                                    ),
                                                    dbc.Tooltip(
                                                        "Como ordenar as categorias (gráfico inferior): Alfabeticamente, Frequência, ou Impacto Shap", # Traduzido
                                                        target="interaction-dependence-bottom-categories-sort-label-"
                                                        + self.name,
                                                    ),
                                                    dbc.Select(
                                                        id="interaction-dependence-bottom-categories-sort-"
                                                        + self.name,
                                                        options=[
                                                            {"label": "Alfabeticamente", "value": "alphabet"},
                                                            {"label": "Frequência", "value": "freq"},
                                                            {"label": "Impacto Shap", "value": "shap"},
                                                        ],
                                                        value=self.cats_sort_bottom,
                                                        size="sm",
                                                    ),
                                                ],
                                                id="interaction-dependence-bottom-categories-div2-"
                                                + self.name,
                                                style=style_bottom_cats, # Estilo original
                                            ),
                                        ],
                                        md=4,
                                    ),
                                    hide=self.hide_cats_sort,
                                ),
                            ]
                        ),
                    ]
                ),
            ],
            class_name="h-100",
        )

    def to_html(self, state_dict=None, add_header=True):
        # Lógica original, sem tradução de eixos/colunas
        args = self.get_state_args(state_dict)
        html_content = "" # Inicializa
        if not hasattr(self.explainer, 'shap_interaction_values') or self.explainer.shap_interaction_values is None:
             html_content = html.Div("Valores de interação SHAP não calculados. Por favor, calcule-os primeiro.") # Traduzido
        elif args["col"] is None or args["interact_col"] is None:
             html_content = html.Div("Por favor, selecione a Característica e a Interação.") # Traduzido
        else:
            fig_top = self.explainer.plot_interaction(
                args["interact_col"], # Y axis feature
                args["col"],          # X axis feature
                highlight_index=args["index"],
                pos_label=args["pos_label"],
                topx=args["cats_topx"],
                sort=args["cats_sort"],
                max_cat_colors=self.max_cat_colors,
                plot_sample=self.plot_sample,
                remove_outliers=bool(args["remove_outliers"]),
            )
            fig_bottom = self.explainer.plot_interaction(
                args["col"],          # Y axis feature
                args["interact_col"], # X axis feature
                highlight_index=args["index"],
                pos_label=args["pos_label"],
                topx=args["cats_topx_bottom"], # Usa valor _bottom
                sort=args["cats_sort_bottom"], # Usa valor _bottom
                max_cat_colors=self.max_cat_colors,
                plot_sample=self.plot_sample,
                remove_outliers=bool(args["remove_outliers_bottom"]), # Usa valor _bottom
            )
            html_content = to_html.fig(fig_top) + html.Hr() + to_html.fig(fig_bottom)

        html_output = to_html.card(html_content, title=self.title, subtitle=self.subtitle)
        if add_header:
            return to_html.add_header(html_output)
        return html_output

    def component_callbacks(self, app):
        # Lógica original dos callbacks
        @app.callback(
            Output("interaction-dependence-interact-col-" + self.name, "options"),
            [
                Input("interaction-dependence-col-" + self.name, "value"),
                Input("pos-label-" + self.name, "value"),
            ],
            # Removido State para manter estrutura original
        )
        def update_interaction_dependence_interact_col(
            col, pos_label # Removido old_interact_col
        ):
            if col is not None:
                if not hasattr(self.explainer, 'shap_interaction_values') or self.explainer.shap_interaction_values is None:
                     return []

                new_interact_cols = self.explainer.top_shap_interactions(
                    col, pos_label=pos_label
                )
                new_interact_cols = [c for c in new_interact_cols if c != col] # Filtra a própria coluna
                new_interact_options = [
                    {"label": c, "value": c} for c in new_interact_cols
                ]
                return new_interact_options
            raise PreventUpdate

        @app.callback(
            [
                Output("interaction-dependence-top-graph-" + self.name, "figure"),
                Output(
                    "interaction-dependence-top-categories-div1-" + self.name, "style"
                ),
                Output(
                    "interaction-dependence-top-categories-div2-" + self.name, "style"
                ),
            ],
            [
                Input("interaction-dependence-interact-col-" + self.name, "value"),
                Input("interaction-dependence-index-" + self.name, "value"),
                Input("interaction-dependence-top-n-categories-" + self.name, "value"),
                Input(
                    "interaction-dependence-top-categories-sort-" + self.name, "value"
                ),
                Input("interaction-dependence-top-outliers-" + self.name, "value"),
                Input("pos-label-" + self.name, "value"),
                Input("interaction-dependence-col-" + self.name, "value"), # Mantido como Input
            ],
        )
        def update_top_dependence_graph( # Nome mantido
            interact_col, index, topx, sort, remove_outliers, pos_label, col
        ):
            if not hasattr(self.explainer, 'shap_interaction_values') or self.explainer.shap_interaction_values is None:
                 return go.Figure(), dict(display="none"), dict(display="none")

            if col is not None and interact_col is not None:
                style = dict(display="none")
                try:
                    if hasattr(self.explainer, 'cat_cols') and interact_col in self.explainer.cat_cols:
                         style = {}
                except AttributeError: pass

                fig = self.explainer.plot_interaction(
                        interact_col, # Y axis feature
                        col,          # X axis feature
                        highlight_index=index,
                        pos_label=pos_label,
                        topx=topx,
                        sort=sort,
                        max_cat_colors=self.max_cat_colors,
                        plot_sample=self.plot_sample,
                        remove_outliers=bool(remove_outliers),
                    )
                return (fig, style, style)
            return go.Figure(), dict(display="none"), dict(display="none") # Retorna vazio se col ou interact_col for None

        @app.callback(
            [
                Output("interaction-dependence-bottom-graph-" + self.name, "figure"),
                Output(
                    "interaction-dependence-bottom-categories-div1-" + self.name,
                    "style",
                ),
                Output(
                    "interaction-dependence-bottom-categories-div2-" + self.name,
                    "style",
                ),
            ],
            [
                Input("interaction-dependence-interact-col-" + self.name, "value"),
                Input("interaction-dependence-index-" + self.name, "value"),
                Input(
                    "interaction-dependence-bottom-n-categories-" + self.name, "value"
                ),
                Input(
                    "interaction-dependence-bottom-categories-sort-" + self.name,
                    "value",
                ),
                Input("interaction-dependence-bottom-outliers-" + self.name, "value"),
                Input("pos-label-" + self.name, "value"),
                Input("interaction-dependence-col-" + self.name, "value"), # Mantido como Input
            ],
        )
        def update_bottom_dependence_graph( # Nome mantido
            interact_col, index, topx, sort, remove_outliers, pos_label, col
        ):
            if not hasattr(self.explainer, 'shap_interaction_values') or self.explainer.shap_interaction_values is None:
                 return go.Figure(), dict(display="none"), dict(display="none")

            if col is not None and interact_col is not None:
                style = dict(display="none")
                try:
                    if hasattr(self.explainer, 'cat_cols') and col in self.explainer.cat_cols:
                         style = {}
                except AttributeError: pass

                fig = self.explainer.plot_interaction(
                        col,          # Y axis feature
                        interact_col, # X axis feature
                        highlight_index=index,
                        pos_label=pos_label,
                        topx=topx, # Usa valor _bottom
                        sort=sort, # Usa valor _bottom
                        max_cat_colors=self.max_cat_colors,
                        plot_sample=self.plot_sample,
                        remove_outliers=bool(remove_outliers), # Usa valor _bottom
                    )
                return (fig, style, style)
            return go.Figure(), dict(display="none"), dict(display="none") # Retorna vazio se col ou interact_col for None


class InteractionSummaryDependenceConnector(ExplainerComponent):
    # Sem interface de utilizador, sem tradução necessária.
    def __init__(self, interaction_summary_component, interaction_dependence_component):
        """Connects a InteractionSummaryComponent with an InteractionDependenceComponent:

        - When select feature in summary, then select col in Dependence
        - When clicking on interaction feature in Summary, then select that interaction
            feature in Dependence.

        Args:
            interaction_summary_component (InteractionSummaryComponent): InteractionSummaryComponent
            interaction_dependence_component (InteractionDependenceComponent): InteractionDependenceComponent
        """
        self.sum_name = interaction_summary_component.name
        self.dep_name = interaction_dependence_component.name

    def component_callbacks(self, app):
        # Lógica original do callback
        @app.callback(
            [
                Output("interaction-dependence-col-" + self.dep_name, "value"),
                Output("interaction-dependence-index-" + self.dep_name, "value"),
                Output("interaction-dependence-interact-col-" + self.dep_name, "value"),
            ],
            [
                Input("interaction-summary-col-" + self.sum_name, "value"),
                Input("interaction-summary-graph-" + self.sum_name, "clickData"),
            ],
             prevent_initial_call=True # Mantido
        )
        def update_interact_col_highlight(col, clickdata):
            index_update = dash.no_update
            interact_col_update = dash.no_update

            if clickdata is not None and clickdata["points"]: # Verifica points
                point_data = clickdata["points"][0]
                if point_data is not None:
                    if isinstance(point_data.get("y"), float):  # detailed
                        try:
                            text_parts = point_data.get("text", "").split("<br>")
                            if len(text_parts) >= 2:
                                index_part = text_parts[0].split("=")
                                interact_col_part = text_parts[1].split("=") # Assume Interaction=col_name
                                if len(index_part) > 1 and len(interact_col_part) > 1:
                                    index = index_part[1].strip()
                                    interact_col = interact_col_part[1].strip()
                                    if self.explainer.index_exists(index) and interact_col in self.explainer.columns:
                                        index_update = index
                                        interact_col_update = interact_col
                        except Exception as e:
                            print(f"Erro ao processar clickData (connector interaction detailed): {e}")
                            pass
                    elif isinstance(point_data.get("y"), str):  # aggregate
                        try:
                            interact_col = point_data["y"].split(":")[-1].strip() # Tenta obter após ':'
                            if interact_col in self.explainer.columns:
                                interact_col_update = interact_col
                        except Exception as e:
                            print(f"Erro ao processar clickData (connector interaction aggregate): {e}")
                            pass

            return (col, index_update, interact_col_update)


class ShapContributionsGraphComponent(ExplainerComponent):
    _state_props = dict(
        index=("contributions-graph-index-", "value"),
        depth=("contributions-graph-depth-", "value"),
        sort=("contributions-graph-sorting-", "value"),
        orientation=("contributions-graph-orientation-", "value"),
        pos_label=("pos-label-", "value"),
    )

    def __init__(
        self,
        explainer,
        title="Gráfico de Contribuições", # Traduzido
        name=None,
        subtitle="Como cada característica contribuiu para a previsão?", # Traduzido
        hide_title=False,
        hide_subtitle=False,
        hide_index=False,
        hide_depth=False,
        hide_sort=False,
        hide_orientation=True, # Mantido True
        hide_selector=False,
        hide_popout=False,
        feature_input_component=None,
        index_dropdown=True,
        pos_label=None,
        index=None,
        depth=None,
        sort="high-to-low",
        orientation="vertical",
        higher_is_better=True,
        description=None,
        **kwargs,
    ):
        """Componente de gráfico de contribuições Shap para previsão

        Args:
            explainer (Explainer): objeto explainer construído com
                        ClassifierExplainer() ou RegressionExplainer()
            title (str, optional): Título do separador ou página. Predefinição para
                        "Gráfico de Contribuições".
            name (str, optional): nome único a adicionar aos elementos do Componente.
                        Se None, um uuid aleatório é gerado para garantir
                        que é único. Predefinição para None.
            subtitle (str): subtítulo
            hide_title (bool, optional): Ocultar título do componente. Predefinição para False.
            hide_subtitle (bool, optional): Ocultar subtítulo. Predefinição para False.
            hide_index (bool, optional): Ocultar seletor de índice. Predefinição para False.
            hide_depth (bool, optional): Ocultar seletor de profundidade. Predefinição para False.
            hide_sort (bool, optional): Ocultar a lista pendente de ordenação. Predefinição para False.
            hide_orientation (bool, optional): Ocultar a lista pendente de orientação.
                    Predefinição para True.
            hide_selector (bool, optional): ocultar seletor de rótulo positivo. Predefinição para False.
            hide_popout (bool, optional): ocultar botão popout
            feature_input_component (FeatureInputComponent): Um FeatureInputComponent
                que fornecerá a entrada para o gráfico em vez do seletor de índice.
                Se não for None, hide_index=True. Predefinição para None.
            index_dropdown (bool, optional): Usar lista pendente para entrada de índice em vez
                de entrada de texto livre. Predefinição para True.
            pos_label ({int, str}, optional): rótulo positivo inicial.
                        Predefinição para explainer.pos_label
            index ({int, bool}, optional): Índice inicial a exibir. Predefinição para None. # Mantido tipo original
            depth (int, optional): Número inicial de características a exibir. Predefinição para None.
            sort ({'abs', 'high-to-low', 'low-to-high', 'importance'}, optional): ordenação dos valores shap.
                        Predefinição para 'high-to-low'.
            orientation ({'vertical', 'horizontal'}, optional): orientação do gráfico de barras.
                        Predefinição para 'vertical'.
            higher_is_better (bool, optional): Colorir valores shap positivos a verde e
                negativos a vermelho, ou o inverso.
            description (str, optional): Dica a exibir ao passar o rato sobre
                o título do componente. Quando None, o texto predefinido é mostrado.
        """
        super().__init__(explainer, title, name)

        self.index_name = "contributions-graph-index-" + self.name

        # Lógica original para depth
        if self.depth is not None:
            self.depth = min(self.depth, self.explainer.n_features)
        # else: # Mantém None

        # Lógica original para feature_input_component
        if self.feature_input_component is not None:
            self.exclude_callbacks(self.feature_input_component)
            self.hide_index = True

        if self.description is None:
            self.description = """
        Este gráfico mostra a contribuição que cada característica individual teve
        na previsão para uma observação específica. As contribuições (começando
        da média da população) somam-se até à previsão final. Isto permite explicar
        exatamente como cada previsão individual foi construída
        a partir de todos os ingredientes individuais no modelo.
        """ # Traduzido

        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)
        self.index_selector = IndexSelector(
            explainer,
            "contributions-graph-index-" + self.name,
            index=index,
            index_dropdown=index_dropdown,
            **kwargs,
        )

        self.popout = GraphPopout(
            "contributions-graph-" + self.name + "popout",
            "contributions-graph-" + self.name,
            self.title,
            self.description,
        )
        self.register_dependencies("shap_values_df")

    def layout(self):
        # Lógica original para opções de profundidade
        depth_options = [{"label": str(i + 1), "value": i + 1} for i in range(self.explainer.n_features)]

        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        self.title, # Traduzido no init
                                        id="contributions-graph-title-" + self.name,
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle" # Traduzido no init
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description, # Traduzido no init
                                        target="contributions-graph-title-" + self.name,
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
                                    dbc.Col([self.selector.layout()], md=2),
                                    hide=self.hide_selector,
                                ),
                            ],
                            justify="end", # Mantido justify original
                        ),
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                f"{self.explainer.index_name}:", # Mantém variável
                                                id="contributions-graph-index-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                f"Selecione o {self.explainer.index_name} para exibir as contribuições das características", # Traduzido
                                                target="contributions-graph-index-label-"
                                                + self.name,
                                            ),
                                            self.index_selector.layout(),
                                        ],
                                        md=4,
                                    ),
                                    hide=self.hide_index,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                "Profundidade:", # Traduzido
                                                id="contributions-graph-depth-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                "Número de características a exibir", # Traduzido
                                                target="contributions-graph-depth-label-"
                                                + self.name,
                                            ),
                                            dbc.Select(
                                                id="contributions-graph-depth-"
                                                + self.name,
                                                # Lógica original para valor e opções
                                                options=[{'label': 'Todas', 'value': 'None'}] + depth_options, # Traduzido 'All'
                                                value='None' if self.depth is None else str(self.depth), # Usa 'None' como string
                                                size="sm",
                                            ),
                                        ],
                                        md=2,
                                    ),
                                    hide=self.hide_depth,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                "Ordenação:", # Traduzido
                                                id="contributions-graph-sorting-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                "Ordenar as características pelo maior impacto absoluto (Absoluto), "
                                                "do mais positivo ao mais negativo (Alto para Baixo), "
                                                "do mais negativo ao mais positivo (Baixo para Alto) ou "
                                                "pela ordenação global de importância (Importância).", # Traduzido
                                                target="contributions-graph-sorting-label-"
                                                + self.name,
                                            ),
                                            dbc.Select(
                                                id="contributions-graph-sorting-"
                                                + self.name,
                                                options=[
                                                    {
                                                        "label": "Absoluto", # Traduzido
                                                        "value": "abs",
                                                    },
                                                    {
                                                        "label": "Alto para Baixo", # Traduzido
                                                        "value": "high-to-low",
                                                    },
                                                    {
                                                        "label": "Baixo para Alto", # Traduzido
                                                        "value": "low-to-high",
                                                    },
                                                    {
                                                        "label": "Importância", # Traduzido
                                                        "value": "importance",
                                                    },
                                                ],
                                                value=self.sort,
                                                size="sm",
                                            ),
                                        ],
                                        md=2,
                                    ),
                                    hide=self.hide_sort,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                "Orientação:", # Traduzido
                                                id="contributions-graph-orientation-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                "Mostrar barras verticais da esquerda para a direita ou barras horizontais de cima para baixo", # Traduzido
                                                target="contributions-graph-orientation-label-"
                                                + self.name,
                                            ),
                                            dbc.Select(
                                                id="contributions-graph-orientation-"
                                                + self.name,
                                                options=[
                                                    {
                                                        "label": "Vertical", # Traduzido
                                                        "value": "vertical",
                                                    },
                                                    {
                                                        "label": "Horizontal", # Traduzido
                                                        "value": "horizontal",
                                                    },
                                                ],
                                                value=self.orientation,
                                                size="sm",
                                            ),
                                        ],
                                        md=2,
                                    ),
                                    hide=self.hide_orientation,
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dcc.Loading(
                                            id="loading-contributions-graph-"
                                            + self.name,
                                            children=[
                                                dcc.Graph(
                                                    id="contributions-graph-"
                                                    + self.name,
                                                    config=dict(
                                                        modeBarButtons=[["toImage"]],
                                                        displaylogo=False,
                                                    ),
                                                )
                                            ],
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

    def get_state_tuples(self):
        # Lógica original
        _state_tuples = super().get_state_tuples()
        if self.feature_input_component is not None:
            _state_tuples.extend(self.feature_input_component.get_state_tuples())
        return sorted(list(set(_state_tuples)))

    def to_html(self, state_dict=None, add_header=True):
        # Lógica original, sem tradução de eixos/colunas
        args = self.get_state_args(state_dict)
        depth_val = None if args["depth"] == 'None' else int(args["depth"])

        html_content = "" # Inicializa
        if self.feature_input_component is None:
            if args["index"] is not None and self.explainer.index_exists(args["index"]):
                fig = self.explainer.plot_contributions(
                    args["index"],
                    topx=depth_val,
                    sort=args["sort"],
                    orientation=args["orientation"],
                    pos_label=args["pos_label"],
                    higher_is_better=self.higher_is_better,
                )
                html_content = to_html.fig(fig)
            else:
                html_content = html.Div("Nenhum índice selecionado ou índice inválido") # Traduzido
        else:
            inputs = {
                k: v
                for k, v in self.feature_input_component.get_state_args(
                    state_dict
                ).items()
                if k != "index"
            }
            inputs_list = list(inputs.values())
            if len(inputs_list) == len(self.feature_input_component._input_features) and not any(i is None for i in inputs_list):
                try:
                    X_row = self.explainer.get_row_from_input(inputs_list, ranked_by_shap=True)
                    if X_row is not None and not X_row.empty:
                        fig = self.explainer.plot_contributions(
                            X_row=X_row,
                            topx=depth_val,
                            sort=args["sort"],
                            orientation=args["orientation"],
                            pos_label=args["pos_label"],
                            higher_is_better=self.higher_is_better,
                        )
                        html_content = to_html.fig(fig)
                    else:
                         html_content = html.Div("Não foi possível gerar a linha de dados a partir da entrada.") # Traduzido
                except Exception as e:
                     print(f"Erro ao gerar gráfico de contribuições a partir da entrada (to_html): {e}")
                     html_content = html.Div("Erro ao processar a entrada.") # Traduzido
            else:
                html_content = html.Div("Dados de entrada incompletos ou incorretos") # Traduzido

        html_output = to_html.card(html_content, title=self.title, subtitle=self.subtitle)
        if add_header:
            return to_html.add_header(html_output)
        return html_output

    def component_callbacks(self, app):
        # Lógica original dos callbacks
        if self.feature_input_component is None:
            @app.callback(
                Output("contributions-graph-" + self.name, "figure"),
                [
                    Input("contributions-graph-index-" + self.name, "value"),
                    Input("contributions-graph-depth-" + self.name, "value"),
                    Input("contributions-graph-sorting-" + self.name, "value"),
                    Input("contributions-graph-orientation-" + self.name, "value"),
                    Input("pos-label-" + self.name, "value"),
                ],
            )
            def update_output_div(index, depth, sort, orientation, pos_label):
                if index is None or not self.explainer.index_exists(index):
                    return go.Figure() # Retorna figura vazia
                depth_val = None if depth == 'None' else int(depth)
                plot = self.explainer.plot_contributions(
                    str(index),
                    topx=depth_val,
                    sort=sort,
                    orientation=orientation,
                    pos_label=pos_label,
                    higher_is_better=self.higher_is_better,
                )
                return plot
        else:
            @app.callback(
                Output("contributions-graph-" + self.name, "figure"),
                [
                    Input("contributions-graph-depth-" + self.name, "value"),
                    Input("contributions-graph-sorting-" + self.name, "value"),
                    Input("contributions-graph-orientation-" + self.name, "value"),
                    Input("pos-label-" + self.name, "value"),
                    *self.feature_input_component._feature_callback_inputs,
                ],
            )
            def update_output_div(depth, sort, orientation, pos_label, *inputs):
                depth_val = None if depth == 'None' else int(depth)
                if not any(i is None for i in inputs):
                    try:
                        X_row = self.explainer.get_row_from_input(
                            inputs, ranked_by_shap=True
                        )
                        if X_row is not None and not X_row.empty:
                            plot = self.explainer.plot_contributions(
                                X_row=X_row,
                                topx=depth_val,
                                sort=sort,
                                orientation=orientation,
                                pos_label=pos_label,
                                higher_is_better=self.higher_is_better,
                            )
                            return plot
                    except Exception as e:
                        print(f"Erro ao gerar gráfico de contribuições a partir da entrada (callback): {e}")
                        return go.Figure() # Retorna figura vazia em caso de erro
                return go.Figure() # Retorna figura vazia se inputs incompletos


class ShapContributionsTableComponent(ExplainerComponent):
    _state_props = dict(
        index=("contributions-table-index-", "value"),
        depth=("contributions-table-depth-", "value"),
        sort=("contributions-table-sorting-", "value"),
        pos_label=("pos-label-", "value"),
    )

    def __init__(
        self,
        explainer,
        title="Tabela de Contribuições", # Traduzido
        name=None,
        subtitle="Como cada característica contribuiu para a previsão?", # Traduzido
        hide_title=False,
        hide_subtitle=False,
        hide_index=False,
        hide_depth=False,
        hide_sort=False,
        hide_selector=False,
        feature_input_component=None,
        index_dropdown=True,
        pos_label=None,
        index=None,
        depth=None,
        sort="abs",
        description=None,
        **kwargs,
    ):
        """Componente que mostra as contribuições dos valores SHAP para a previsão numa tabela

        Args:
            explainer (Explainer): objeto explainer construído com
                        ClassifierExplainer() ou RegressionExplainer()
            title (str, optional): Título do separador ou página. Predefinição para
                        "Tabela de Contribuições".
            name (str, optional): nome único a adicionar aos elementos do Componente.
                        Se None, um uuid aleatório é gerado para garantir
                        que é único. Predefinição para None.
            subtitle (str): subtítulo
            hide_title (bool, optional): Ocultar título do componente. Predefinição para False.
            hide_subtitle (bool, optional): Ocultar subtítulo. Predefinição para False.
            hide_index (bool, optional): Ocultar seletor de índice. Predefinição para False.
            hide_depth (bool, optional): Ocultar seletor de profundidade. Predefinição para False.
            hide_sort (bool, optional): Ocultar lista pendente de ordenação. Predefinição para False.
            hide_selector (bool, optional): ocultar seletor de rótulo positivo. Predefinição para False.
            feature_input_component (FeatureInputComponent): Um FeatureInputComponent
                que fornecerá a entrada para o gráfico em vez do seletor de índice.
                Se não for None, hide_index=True. Predefinição para None.
            index_dropdown (bool, optional): Usar lista pendente para entrada de índice em vez
                de entrada de texto livre. Predefinição para True.
            pos_label ({int, str}, optional): rótulo positivo inicial.
                        Predefinição para explainer.pos_label
            index ([type], optional): Índice inicial a exibir. Predefinição para None.
            depth ([type], optional): Número inicial de características a exibir. Predefinição para None.
            sort ({'abs', 'high-to-low', 'low-to-high', 'importance'}, optional): ordenação dos valores shap.
                        Predefinição para 'abs'.
            description (str, optional): Dica a exibir ao passar o rato sobre
                o título do componente. Quando None, o texto predefinido é mostrado.
        """
        super().__init__(explainer, title, name)

        self.index_name = "contributions-table-index-" + self.name

        # Lógica original para depth
        if self.depth is not None:
            self.depth = min(self.depth, self.explainer.n_features)
        # else: # Mantém None

        # Lógica original para feature_input_component
        if self.feature_input_component is not None:
            self.exclude_callbacks(self.feature_input_component)
            self.hide_index = True

        if self.description is None:
            self.description = """
        Esta tabela mostra a contribuição que cada característica individual teve
        na previsão para uma observação específica. As contribuições (começando
        da média da população) somam-se até à previsão final. Isto permite explicar
        exatamente como cada previsão individual foi construída
        a partir de todos os ingredientes individuais no modelo.
        """ # Traduzido
        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)
        self.index_selector = IndexSelector(
            explainer,
            "contributions-table-index-" + self.name,
            index=index,
            index_dropdown=index_dropdown,
            **kwargs,
        )

        self.register_dependencies("shap_values_df")

    def layout(self):
        # Lógica original para opções de profundidade
        depth_options = [{"label": str(i + 1), "value": i + 1} for i in range(self.explainer.n_features)]

        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        self.title, # Traduzido no init
                                        id="contributions-table-title-" + self.name,
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle" # Traduzido no init
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description, # Traduzido no init
                                        target="contributions-table-title-" + self.name,
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
                                                f"{self.explainer.index_name}:", # Mantém variável
                                                id="contributions-table-index-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                f"Selecione o {self.explainer.index_name} para exibir as contribuições das características", # Traduzido
                                                target="contributions-table-index-label-"
                                                + self.name,
                                            ),
                                            self.index_selector.layout(),
                                        ],
                                        md=4,
                                    ),
                                    hide=self.hide_index,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                "Profundidade:", # Traduzido
                                                id="contributions-table-depth-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                "Número de características a exibir", # Traduzido
                                                target="contributions-table-depth-label-"
                                                + self.name,
                                            ),
                                            dbc.Select(
                                                id="contributions-table-depth-"
                                                + self.name,
                                                # Lógica original para valor e opções
                                                options=[{'label': 'Todas', 'value': 'None'}] + depth_options, # Traduzido 'All'
                                                value='None' if self.depth is None else str(self.depth), # Usa 'None' como string
                                                size="sm",
                                            ),
                                        ],
                                        md=2,
                                    ),
                                    hide=self.hide_depth,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                "Ordenação:", # Traduzido
                                                id="contributions-table-sorting-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                "Ordenar as características pelo maior impacto absoluto (Absoluto), "
                                                "do mais positivo ao mais negativo (Alto para Baixo), "
                                                "do mais negativo ao mais positivo (Baixo para Alto) ou "
                                                "pela ordenação global de importância (Importância).", # Traduzido
                                                target="contributions-table-sorting-label-"
                                                + self.name,
                                            ),
                                            dbc.Select(
                                                id="contributions-table-sorting-"
                                                + self.name,
                                                options=[
                                                    {"label": "Absoluto", "value": "abs"},
                                                    {"label": "Alto para Baixo", "value": "high-to-low"},
                                                    {"label": "Baixo para Alto", "value": "low-to-high"},
                                                    {"label": "Importância", "value": "importance"},
                                                ],
                                                value=self.sort,
                                                size="sm",
                                            ),
                                        ],
                                        md=2,
                                    ),
                                    hide=self.hide_sort,
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
                                            id="loading-contributions-table-"
                                            + self.name,
                                            children=[
                                                html.Div(
                                                    id="contributions-table-"
                                                    + self.name
                                                )
                                            ],
                                        ),
                                    ]
                                ),
                            ]
                        ),
                    ]
                ),
            ],
            class_name="h-100",
        )

    def get_state_tuples(self):
        # Lógica original
        _state_tuples = super().get_state_tuples()
        if self.feature_input_component is not None:
            _state_tuples.extend(self.feature_input_component.get_state_tuples())
        return sorted(list(set(_state_tuples)))

    def to_html(self, state_dict=None, add_header=True):
        # Lógica original, sem tradução de colunas
        args = self.get_state_args(state_dict)
        depth_val = None if args["depth"] == 'None' else int(args["depth"])

        html_content = "" # Inicializa
        if self.feature_input_component is None:
            if args["index"] is not None and self.explainer.index_exists(args["index"]):
                contrib_df = self.explainer.get_contrib_summary_df(
                    args["index"],
                    topx=depth_val,
                    sort=args["sort"],
                    pos_label=args["pos_label"],
                )
                html_content = to_html.table_from_df(contrib_df)
            else:
                html_content = html.Div("Nenhum índice selecionado ou índice inválido") # Traduzido
        else:
            inputs = {
                k: v
                for k, v in self.feature_input_component.get_state_args(
                    state_dict
                ).items()
                if k != "index"
            }
            inputs_list = list(inputs.values())
            if len(inputs_list) == len(self.feature_input_component._input_features) and not any(i is None for i in inputs_list):
                try:
                    X_row = self.explainer.get_row_from_input(inputs_list, ranked_by_shap=True)
                    if X_row is not None and not X_row.empty:
                        contrib_df = self.explainer.get_contrib_summary_df(
                            X_row=X_row,
                            topx=depth_val,
                            sort=args["sort"],
                            pos_label=args["pos_label"],
                        )
                        html_content = to_html.table_from_df(contrib_df)
                    else:
                        html_content = html.Div("Não foi possível gerar a linha de dados a partir da entrada.") # Traduzido
                except Exception as e:
                    print(f"Erro ao gerar tabela de contribuições a partir da entrada (to_html): {e}")
                    html_content = html.Div("Erro ao processar a entrada.") # Traduzido
            else:
                html_content = html.Div("Dados de entrada incompletos ou incorretos") # Traduzido

        html_output = to_html.card(html_content, title=self.title, subtitle=self.subtitle)
        if add_header:
            return to_html.add_header(html_output)
        return html_output

    def component_callbacks(self, app):
        # Lógica original dos callbacks
        if self.feature_input_component is None:
            @app.callback(
                Output("contributions-table-" + self.name, "children"),
                [
                    Input("contributions-table-index-" + self.name, "value"),
                    Input("contributions-table-depth-" + self.name, "value"),
                    Input("contributions-table-sorting-" + self.name, "value"),
                    Input("pos-label-" + self.name, "value"),
                ],
            )
            def update_output_div(index, depth, sort, pos_label):
                if index is None or not self.explainer.index_exists(index):
                    return html.Div("Por favor, selecione um índice válido.") # Traduzido
                depth_val = None if depth == 'None' else int(depth)
                contrib_df = self.explainer.get_contrib_summary_df(
                        str(index), topx=depth_val, sort=sort, pos_label=pos_label
                    )
                # Gera tabela sem traduzir colunas
                contributions_table = dbc.Table.from_dataframe(contrib_df, striped=True, bordered=True, hover=True)

                # Lógica original para tooltips
                tooltip_cols = {}
                if hasattr(contributions_table, 'children') and len(contributions_table.children) > 1:
                    tbody = contributions_table.children[1]
                    if hasattr(tbody, 'children'):
                        for i, tr in enumerate(tbody.children): # Adicionado enumerate para ID único
                            try:
                                tds = tr.children
                                col = tds[0].children.split(" = ")[0].strip() # Assume formato 'Feature = Value'
                                desc = self.explainer.description(col)
                                if desc != "":
                                    row_id = f"contributions-table-hover-{self.name}-{i}" # ID único
                                    tr.id = row_id
                                    tooltip_cols[row_id] = desc # Usa ID da linha como target
                            except Exception as e:
                                print(f"Erro ao processar linha da tabela para tooltip (callback index): {e}")
                                pass

                tooltips = [
                    dbc.Tooltip(
                        desc,
                        target=row_id, # Usa ID da linha
                        placement="top",
                    )
                    for row_id, desc in tooltip_cols.items()
                ]

                output_div = html.Div([contributions_table, *tooltips])
                return output_div
        else:
            @app.callback(
                Output("contributions-table-" + self.name, "children"),
                [
                    Input("contributions-table-depth-" + self.name, "value"),
                    Input("contributions-table-sorting-" + self.name, "value"),
                    Input("pos-label-" + self.name, "value"),
                    *self.feature_input_component._feature_callback_inputs,
                ],
            )
            def update_output_div(depth, sort, pos_label, *inputs):
                depth_val = None if depth == 'None' else int(depth)
                if not any(i is None for i in inputs):
                    try:
                        X_row = self.explainer.get_row_from_input(
                            inputs, ranked_by_shap=True
                        )
                        if X_row is not None and not X_row.empty:
                            contrib_df = self.explainer.get_contrib_summary_df(
                                X_row=X_row, topx=depth_val, sort=sort, pos_label=pos_label
                            )
                            # Gera tabela sem traduzir colunas
                            contributions_table = dbc.Table.from_dataframe(contrib_df, striped=True, bordered=True, hover=True)

                            # Lógica original para tooltips
                            tooltip_cols = {}
                            if hasattr(contributions_table, 'children') and len(contributions_table.children) > 1:
                                tbody = contributions_table.children[1]
                                if hasattr(tbody, 'children'):
                                    for i, tr in enumerate(tbody.children): # Adicionado enumerate
                                        try:
                                            tds = tr.children
                                            col = tds[0].children.split(" = ")[0].strip()
                                            desc = self.explainer.description(col)
                                            if desc != "":
                                                row_id = f"contributions-table-hover-{self.name}-{i}" # ID único
                                                tr.id = row_id
                                                tooltip_cols[row_id] = desc # Usa ID da linha
                                        except Exception as e:
                                            print(f"Erro ao processar linha da tabela para tooltip (callback input): {e}")
                                            pass

                            tooltips = [
                                dbc.Tooltip(
                                    desc,
                                    target=row_id, # Usa ID da linha
                                    placement="top",
                                )
                                for row_id, desc in tooltip_cols.items()
                            ]

                            output_div = html.Div([contributions_table, *tooltips])
                            return output_div
                        else:
                            return html.Div("Não foi possível gerar a linha de dados a partir da entrada.") # Traduzido
                    except Exception as e:
                        print(f"Erro ao gerar tabela de contribuições a partir da entrada (callback): {e}")
                        return html.Div("Erro ao processar a entrada.") # Traduzido
                return html.Div("Por favor, forneça todos os dados de entrada.") # Traduzido