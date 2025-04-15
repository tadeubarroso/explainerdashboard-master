# -*- coding: utf-8 -*-

__all__ = [
    "CutoffPercentileComponent",
    "PosLabelConnector",
    "CutoffConnector",
    "IndexConnector",
    "HighlightConnector",
]

import numpy as np

import dash
from dash import html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

# Presumo que os imports abaixo estejam corretos no contexto do projeto original
from ..dashboard_methods import *


class CutoffPercentileComponent(ExplainerComponent):
    def __init__(
        self,
        explainer,
        title="Cutoff Global", # Traduzido
        name=None,
        hide_title=False,
        hide_cutoff=False,
        hide_percentile=False,
        hide_selector=False,
        pos_label=None,
        cutoff=0.5,
        percentile=None,
        description=None,
        **kwargs,
    ):
        """
        Slider to set a cutoff for Classifier components, based on setting the
        cutoff at a certain percentile of predictions, e.g.:
        percentile=0.8 means "mark the 20% highest scores as positive".

        This cutoff can then be conencted with other components like e.g.
        RocAucComponent with a CutoffConnector.

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Cutoff Global". # Atualizado docstring
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            hide_title (bool, optional): Hide title.
            hide_cutoff (bool, optional): Hide the cutoff slider. Defaults to False.
            hide_percentile (bool, optional): Hide percentile slider. Defaults to False.
            hide_selector (bool, optional): hide pos label selectors. Defaults to False.
            pos_label ({int, str}, optional): initial pos label.
                        Defaults to explainer.pos_label
            cutoff (float, optional): Initial cutoff. Defaults to 0.5.
            percentile ([type], optional): Initial percentile. Defaults to None.
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown.
        """
        super().__init__(explainer, title, name)

        self.cutoff_name = "cutoffconnector-cutoff-" + self.name

        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)

        if self.description is None:
            # Traduzido
            self.description = """
        Selecione um cutoff do modelo tal que todas as probabilidades previstas
        acima do cutoff sejam rotuladas como positivas, e todas as probabilidades
        previstas abaixo do cutoff sejam rotuladas como negativas. Também pode definir
        o cutoff como um percentil de todas as observações. Definir o cutoff
        aqui irá definir automaticamente o cutoff em múltiplos outros componentes
        conectados.
        """
        self.register_dependencies(["preds", "pred_percentiles"])

    def layout(self):
        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.H3(
                                self.title, # Usa self.title
                                className="card-title",
                                id="cutoffconnector-title-" + self.name,
                            ),
                            dbc.Tooltip(
                                self.description, # Usa self.description
                                target="cutoffconnector-title-" + self.name,
                            ),
                        ]
                    ),
                    hide=self.hide_title,
                ),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Row(
                                            [
                                                make_hideable(
                                                    dbc.Col(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.Label(
                                                                        # Traduzido
                                                                        "Cutoff da probabilidade de previsão:"
                                                                    ),
                                                                    dcc.Slider(
                                                                        id="cutoffconnector-cutoff-"
                                                                        + self.name,
                                                                        min=0.01,
                                                                        max=0.99,
                                                                        step=0.01,
                                                                        value=self.cutoff,
                                                                        marks={
                                                                            0.01: "0.01",
                                                                            0.25: "0.25",
                                                                            0.50: "0.50",
                                                                            0.75: "0.75",
                                                                            0.99: "0.99",
                                                                        },
                                                                        included=False,
                                                                        tooltip={
                                                                            "always_visible": False
                                                                        },
                                                                    ),
                                                                ],
                                                                style={
                                                                    "margin-bottom": 15
                                                                },
                                                                id="cutoffconnector-cutoff-div-"
                                                                + self.name,
                                                            ),
                                                            dbc.Tooltip(
                                                                # Traduzido
                                                                f"Pontuações acima deste cutoff serão rotuladas como positivas",
                                                                target="cutoffconnector-cutoff-div-"
                                                                + self.name,
                                                                placement="bottom",
                                                            ),
                                                        ]
                                                    ),
                                                    hide=self.hide_cutoff,
                                                ),
                                            ]
                                        ),
                                        dbc.Row(
                                            [
                                                make_hideable(
                                                    dbc.Col(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.Label(
                                                                        # Traduzido
                                                                        "Percentil de cutoff das amostras:"
                                                                    ),
                                                                    dcc.Slider(
                                                                        id="cutoffconnector-percentile-"
                                                                        + self.name,
                                                                        min=0.01,
                                                                        max=0.99,
                                                                        step=0.01,
                                                                        value=self.percentile,
                                                                        marks={
                                                                            0.01: "0.01",
                                                                            0.25: "0.25",
                                                                            0.50: "0.50",
                                                                            0.75: "0.75",
                                                                            0.99: "0.99",
                                                                        },
                                                                        included=False,
                                                                        tooltip={
                                                                            "always_visible": False
                                                                        },
                                                                    ),
                                                                ],
                                                                style={
                                                                    "margin-bottom": 15
                                                                },
                                                                id="cutoffconnector-percentile-div-"
                                                                + self.name,
                                                            ),
                                                            dbc.Tooltip(
                                                                # Traduzido
                                                                f"exemplo: se definir para percentil=0.9: rotula as 10% maiores pontuações como positivas, o resto negativas.",
                                                                target="cutoffconnector-percentile-div-"
                                                                + self.name,
                                                                placement="bottom",
                                                            ),
                                                        ]
                                                    ),
                                                    hide=self.hide_percentile,
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                                make_hideable(
                                    dbc.Col([self.selector.layout()], width=2),
                                    hide=self.hide_selector,
                                ),
                            ]
                        )
                    ]
                ),
            ]
        )

    def component_callbacks(self, app):
        @app.callback(
            Output("cutoffconnector-cutoff-" + self.name, "value"),
            [
                Input("cutoffconnector-percentile-" + self.name, "value"),
                Input("pos-label-" + self.name, "value"),
            ],
        )
        def update_cutoff(percentile, pos_label):
            if percentile is not None:
                return np.round(
                    self.explainer.cutoff_from_percentile(
                        percentile, pos_label=pos_label
                    ),
                    2,
                )
            raise PreventUpdate


class PosLabelConnector(ExplainerComponent):
    def __init__(self, input_pos_label, output_pos_labels):
        self.input_pos_label_name = self._get_pos_label(input_pos_label)
        self.output_pos_label_names = self._get_pos_labels(output_pos_labels)
        # if self.input_pos_label_name in self.output_pos_label_names:
        #     # avoid circulat callbacks
        #     self.output_pos_label_names.remove(self.input_pos_label_name)

    def _get_pos_label(self, input_pos_label):
        if isinstance(input_pos_label, PosLabelSelector):
            return "pos-label-" + input_pos_label.name
        elif hasattr(input_pos_label, "selector") and isinstance(
            input_pos_label.selector, PosLabelSelector
        ):
            return "pos-label-" + input_pos_label.selector.name
        elif isinstance(input_pos_label, str):
            return input_pos_label
        else:
            raise ValueError(
                 # Traduzido
                "input_pos_label deve ser str, "
                "PosLabelSelector ou uma instância com propriedade .selector"
                " que seja um PosLabelSelector!"
            )

    def _get_pos_labels(self, output_pos_labels):
        def get_pos_labels(o):
            if isinstance(o, PosLabelSelector):
                return ["pos-label-" + o.name]
            elif isinstance(o, str):
                # Correção: deveria retornar [o] se for string, não [str]
                return [o]
            elif hasattr(o, "pos_labels"):
                return o.pos_labels
            return []

        if hasattr(output_pos_labels, "__iter__"):
            pos_labels = []
            for comp in output_pos_labels:
                pos_labels.extend(get_pos_labels(comp))
            return list(set(pos_labels))
        else:
            return get_pos_labels(output_pos_labels)

    def component_callbacks(self, app):
        if self.output_pos_label_names:

            @app.callback(
                [
                    Output(pos_label_name, "value")
                    for pos_label_name in self.output_pos_label_names
                ],
                [Input(self.input_pos_label_name, "value")],
            )
            def update_pos_labels(pos_label):
                return tuple(pos_label for i in range(len(self.output_pos_label_names)))


class CutoffConnector(ExplainerComponent):
    def __init__(self, input_cutoff, output_cutoffs):
        """Connect the cutoff selector of input_cutoff with those of output_cutoffs.

        You can use this to connect a CutoffPercentileComponent with a
        RocAucComponent for example,

        When you change the cutoff in input_cutoff, all the cutoffs in output_cutoffs
        will automatically be updated.

        Args:
            input_cutoff ([{str, ExplainerComponent}]): Either a str or an
                        ExplainerComponent. If str should be equal to the
                        name of the cutoff property. If ExplainerComponent then
                        should have a .cutoff_name property.
            output_cutoffs (list(str, ExplainerComponent)): list of str of
                        ExplainerComponents.
        """
        self.input_cutoff_name = self.cutoff_name(input_cutoff)
        self.output_cutoff_names = self.cutoff_name(output_cutoffs)
        if not isinstance(self.output_cutoff_names, list):
            self.output_cutoff_names = [self.output_cutoff_names]

    @staticmethod
    def cutoff_name(cutoffs):
        def get_cutoff_name(o):
            if isinstance(o, str):
                return o
            elif isinstance(o, ExplainerComponent):
                if not hasattr(o, "cutoff_name"):
                    raise ValueError(f"{o} não tem uma propriedade .cutoff_name!") # Traduzido
                return o.cutoff_name
            raise ValueError(
                 # Traduzido
                f"{o} não é nem str nem um ExplainerComponent com uma propriedade .cutoff_name"
            )

        if hasattr(cutoffs, "__iter__"):
            cutoff_name_list = []
            for cutoff in cutoffs:
                cutoff_name_list.append(get_cutoff_name(cutoff))
            return cutoff_name_list
        else:
            return get_cutoff_name(cutoffs)

    def component_callbacks(self, app):
        @app.callback(
            [Output(cutoff_name, "value") for cutoff_name in self.output_cutoff_names],
            [Input(self.input_cutoff_name, "value")],
        )
        def update_cutoffs(cutoff):
            return tuple(cutoff for i in range(len(self.output_cutoff_names)))


class IndexConnector(ExplainerComponent):
    def __init__(self, input_index, output_indexes, explainer=None):
        """Connect the index selector of input_index with those of output_indexes.

        You can use this to connect a RandomIndexComponent with a
        PredictionSummaryComponent for example.

        When you change the index in input_index, all the indexes in output_indexes
        will automatically be updated.

        Args:
            input_index ([{str, ExplainerComponent}]): Either a str or an
                        ExplainerComponent. If str should be equal to the
                        name of the index property. If ExplainerComponent then
                        should have a .index_name property.
            output_indexes (list(str, ExplainerComponent)): list of str of
                        ExplainerComponents.
        """
        self.input_index_name = self.index_name(input_index)
        self.output_index_names = self.index_name(output_indexes)
        self.explainer = explainer
        if not isinstance(self.output_index_names, list):
            self.output_index_names = [self.output_index_names]

    @staticmethod
    def index_name(indexes):
        def get_index_name(o):
            if isinstance(o, str):
                return o
            elif isinstance(o, ExplainerComponent):
                if not hasattr(o, "index_name"):
                    raise ValueError(f"{o} não tem uma propriedade .index_name!") # Traduzido
                return o.index_name
            raise ValueError(
                 # Traduzido
                f"{o} não é nem str nem um ExplainerComponent com uma propriedade .index_name"
            )

        if hasattr(indexes, "__iter__"):
            index_name_list = []
            for index in indexes:
                index_name_list.append(get_index_name(index))
            return index_name_list
        else:
            return get_index_name(indexes)

    def component_callbacks(self, app):
        @app.callback(
            [Output(index_name, "value") for index_name in self.output_index_names],
            [Input(self.input_index_name, "value")],
        )
        def update_indexes(index):
            if dash.callback_context.triggered_id != self.input_index_name:
                raise PreventUpdate
            if self.explainer is not None:
                if index is not None and self.explainer.index_exists(index):
                    return tuple([index for _ in self.output_index_names])
            elif index is not None:
                return tuple([index for _ in self.output_index_names])
            raise PreventUpdate


class HighlightConnector(ExplainerComponent):
    def __init__(self, input_highlight, output_highlights):
        """Connect the highlight selector of input_highlight with those of output_highlights.

        You can use this to connect a DecisionTreesComponent component to a
        DecisionPathGraphComponent for example.

        When you change the highlight in input_highlight, all the highlights in output_highlights
        will automatically be updated.

        Args:
            input_highlight ([{str, ExplainerComponent}]): Either a str or an
                        ExplainerComponent. If str should be equal to the
                        name of the highlight property. If ExplainerComponent then
                        should have a .highlight_name property.
            output_highlights (list(str, ExplainerComponent)): list of str of
                        ExplainerComponents.
        """
        self.input_highlight_name = self.highlight_name(input_highlight)
        self.output_highlight_names = self.highlight_name(output_highlights)
        if not isinstance(self.output_highlight_names, list):
            self.output_highlight_names = [self.output_highlight_names]

    @staticmethod
    def highlight_name(highlights):
        def get_highlight_name(o):
            if isinstance(o, str):
                return o
            elif isinstance(o, ExplainerComponent):
                if not hasattr(o, "highlight_name"):
                    raise ValueError(f"{o} não tem uma propriedade .highlight_name!") # Traduzido
                return o.highlight_name
            raise ValueError(
                 # Traduzido
                f"{o} não é nem str nem um ExplainerComponent com uma propriedade .highlight_name"
            )

        if hasattr(highlights, "__iter__"):
            highlight_name_list = []
            for highlight in highlights:
                highlight_name_list.append(get_highlight_name(highlight))
            return highlight_name_list
        else:
            return get_highlight_name(highlights)

    def component_callbacks(self, app):
        @app.callback(
            [
                Output(highlight_name, "value")
                for highlight_name in self.output_highlight_names
            ],
            [Input(self.input_highlight_name, "value")],
        )
        def update_highlights(highlight):
            return tuple(highlight for i in range(len(self.output_highlight_names)))
        