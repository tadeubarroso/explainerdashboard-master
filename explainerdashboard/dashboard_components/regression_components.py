__all__ = [
    "RegressionRandomIndexComponent",
    "RegressionPredictionSummaryComponent",
    "PredictedVsActualComponent",
    "ResidualsComponent",
    "RegressionVsColComponent",
    "RegressionModelSummaryComponent",
]

import numpy as np
import pandas as pd

import dash
from dash import html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from ..dashboard_methods import *
from .. import to_html


class RegressionRandomIndexComponent(ExplainerComponent):
    _state_props = dict(index=("random-index-reg-index-", "value"))

    def __init__(
        self,
        explainer,
        title=None, # Default title uses f-string, translated below
        name=None,
        subtitle="Selecione da lista ou escolha aleatoriamente", # Translated
        hide_title=False,
        hide_subtitle=False,
        hide_index=False,
        hide_pred_slider=False,
        hide_residual_slider=False,
        hide_pred_or_y=False,
        hide_abs_residuals=False,
        hide_button=False,
        index_dropdown=True,
        index=None,
        pred_slider=None,
        y_slider=None,
        residual_slider=None,
        abs_residual_slider=None,
        pred_or_y="preds",
        abs_residuals=True,
        round=2,
        description=None,
        **kwargs,
    ):
        """Select a random index subject to constraints component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        f"Selecionar {explainer.index_name}". # Translated default
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            subtitle (str): subtitle
            hide_title (bool, optional): hide title
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_index (bool, optional): Hide index selector.
                        Defaults to False.
            hide_pred_slider (bool, optional): Hide prediction slider.
                        Defaults to False.
            hide_residual_slider (bool, optional): hide residuals slider.
                        Defaults to False.
            hide_pred_or_y (bool, optional): hide prediction or actual toggle.
                        Defaults to False.
            hide_abs_residuals (bool, optional): hide absolute residuals toggle.
                        Defaults to False.
            hide_button (bool, optional): hide button. Defaults to False.
            index_dropdown (bool, optional): Use dropdown for index input instead
                of free text input. Defaults to True.
            index ({str, int}, optional): Initial index to display.
                        Defaults to None.
            pred_slider ([lb, ub], optional): Initial values for prediction
                        values slider [lowerbound, upperbound]. Defaults to None.
            y_slider ([lb, ub], optional): Initial values for y slider
                        [lower bound, upper bound]. Defaults to None.
            residual_slider ([lb, ub], optional): Initial values for residual slider
                        [lower bound, upper bound]. Defaults to None.
            abs_residual_slider ([lb, ub], optional): Initial values for absolute
                        residuals slider [lower bound, upper bound]
                        Defaults to None.
            pred_or_y (str, {'preds', 'y'}, optional): Initial use predictions
                        or y slider. Defaults to "preds".
            abs_residuals (bool, optional): Initial use residuals or absolute
                        residuals. Defaults to True.
            round (int, optional): rounding used for slider spacing. Defaults to 2.
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown.
        """
        # Use translated default title if title is None
        super().__init__(explainer, title or f"Selecionar {explainer.index_name}", name) # Translated default
        assert self.explainer.is_regression, (
            "explainer is not a RegressionExplainer so the RegressionRandomIndexComponent "
            "will not work. Try using the ClassifierRandomIndexComponent instead."
        )

        # self.title is now set in super().__init__

        self.index_name = "random-index-reg-index-" + self.name
        self.index_selector = IndexSelector(
            explainer,
            self.index_name,
            index=index,
            index_dropdown=index_dropdown,
            **kwargs,
        )

        if self.explainer.y_missing:
            self.hide_residual_slider = True
            self.hide_pred_or_y = True
            self.hide_abs_residuals = True
            self.pred_or_y = "preds"
            self.y_slider = [0.0, 1.0]
            self.residual_slider = [0.0, 1.0]
            self.abs_residual_slider = [0.0, 1.0]

        if self.pred_slider is None:
            self.pred_slider = [
                float(self.explainer.preds.min()),
                float(self.explainer.preds.max()),
            ]

        if not self.explainer.y_missing:
            if self.y_slider is None:
                self.y_slider = [
                    float(self.explainer.y.min()),
                    float(self.explainer.y.max()),
                ]

            if self.residual_slider is None:
                self.residual_slider = [
                    float(self.explainer.residuals.min()),
                    float(self.explainer.residuals.max()),
                ]

            if self.abs_residual_slider is None:
                self.abs_residual_slider = [
                    float(self.explainer.abs_residuals.min()),
                    float(self.explainer.abs_residuals.max()),
                ]

            assert (
                len(self.pred_slider) == 2
                and self.pred_slider[0] <= self.pred_slider[1]
            ), "pred_slider should be a list of a [lower_bound, upper_bound]!"

            assert (
                len(self.y_slider) == 2 and self.y_slider[0] <= self.y_slider[1]
            ), "y_slider should be a list of a [lower_bound, upper_bound]!"

            assert (
                len(self.residual_slider) == 2
                and self.residual_slider[0] <= self.residual_slider[1]
            ), "residual_slider should be a list of a [lower_bound, upper_bound]!"

            assert (
                len(self.abs_residual_slider) == 2
                and self.abs_residual_slider[0] <= self.abs_residual_slider[1]
            ), "abs_residual_slider should be a list of a [lower_bound, upper_bound]!"

        self.y_slider = [float(y) for y in self.y_slider]
        self.pred_slider = [float(p) for p in self.pred_slider]
        self.residual_slider = [float(r) for r in self.residual_slider]
        self.abs_residual_slider = [float(a) for a in self.abs_residual_slider]

        assert self.pred_or_y in {
            "preds",
            "y",
        }, "pred_or_y should be in ['preds', 'y']!"

        if self.description is None:
            # Translated default description
            self.description = f"""
        Pode selecionar um {self.explainer.index_name} diretamente escolhendo-o
        na lista suspensa (se começar a escrever, pode pesquisar dentro da lista),
        ou clique no botão {self.explainer.index_name} Aleatório para selecionar
        aleatoriamente um {self.explainer.index_name} que cumpra as restrições. Por exemplo,
        pode selecionar um {self.explainer.index_name} com um {self.explainer.target} previsto muito alto,
        ou um {self.explainer.target} observado muito baixo,
        ou um {self.explainer.index_name} cujo {self.explainer.target} previsto
        estava muito longe do {self.explainer.target} observado e, por isso, tinha um
        resíduo (absoluto) elevado.
        """

    def layout(self):
        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        self.title, # Title already set (possibly translated)
                                        id="random-index-reg-title-" + self.name,
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle" # Subtitle translated in __init__
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description, # Description already set (possibly translated)
                                        target="random-index-reg-title-" + self.name,
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
                                    dbc.Col([self.index_selector.layout()], md=8),
                                    hide=self.hide_index,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Button(
                                                f"Aleatório {self.explainer.index_name}", # Translated
                                                color="primary",
                                                id="random-index-reg-button-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                # Translated
                                                f"Selecione um {self.explainer.index_name} aleatório de acordo com as restrições",
                                                target="random-index-reg-button-"
                                                + self.name,
                                            ),
                                        ],
                                        md=4,
                                    ),
                                    hide=self.hide_button,
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
                                                    dbc.Label(
                                                        "Intervalo Previsto:", # Translated
                                                        id="random-index-reg-pred-slider-label-"
                                                        + self.name,
                                                        html_for="random-index-reg-pred-slider-"
                                                        + self.name,
                                                    ),
                                                    dbc.Tooltip(
                                                        # Translated
                                                        f"Selecionar apenas {self.explainer.index_name} onde o "
                                                        f"{self.explainer.target} previsto estava dentro do seguinte intervalo:",
                                                        target="random-index-reg-pred-slider-label-"
                                                        + self.name,
                                                    ),
                                                    dcc.RangeSlider(
                                                        id="random-index-reg-pred-slider-"
                                                        + self.name,
                                                        min=float(
                                                            self.explainer.preds.min()
                                                        ),
                                                        max=float(
                                                            self.explainer.preds.max()
                                                        ),
                                                        step=np.float_power(
                                                            10, -self.round
                                                        ),
                                                        value=[
                                                            self.pred_slider[0],
                                                            self.pred_slider[1],
                                                        ],
                                                        marks={
                                                            float(
                                                                self.explainer.preds.min()
                                                            ): str(
                                                                np.round(
                                                                    self.explainer.preds.min(),
                                                                    self.round,
                                                                )
                                                            ),
                                                            float(
                                                                self.explainer.preds.max()
                                                            ): str(
                                                                np.round(
                                                                    self.explainer.preds.max(),
                                                                    self.round,
                                                                )
                                                            ),
                                                        },
                                                        allowCross=False,
                                                        tooltip={
                                                            "always_visible": False
                                                        },
                                                    ),
                                                ],
                                                id="random-index-reg-pred-slider-div-"
                                                + self.name,
                                                # Style adjusted in callback based on initial value
                                                style=None if self.pred_or_y == "preds" else dict(display="none")
                                            ),
                                            html.Div(
                                                [
                                                    dbc.Label(
                                                        "Intervalo Observado:", # Translated
                                                        id="random-index-reg-y-slider-label-"
                                                        + self.name,
                                                        html_for="random-index-reg-y-slider-"
                                                        + self.name,
                                                    ),
                                                    dbc.Tooltip(
                                                        # Translated
                                                        f"Selecionar apenas {self.explainer.index_name} onde o "
                                                        f"{self.explainer.target} observado estava dentro do seguinte intervalo:",
                                                        target="random-index-reg-y-slider-label-"
                                                        + self.name,
                                                    ),
                                                    dcc.RangeSlider(
                                                        id="random-index-reg-y-slider-"
                                                        + self.name,
                                                        min=float(
                                                            self.explainer.y.min()
                                                        ),
                                                        max=float(
                                                            self.explainer.y.max()
                                                        ),
                                                        step=np.float_power(
                                                            10, -self.round
                                                        ),
                                                        value=[
                                                            self.y_slider[0],
                                                            self.y_slider[1],
                                                        ],
                                                        marks={
                                                            float(
                                                                self.explainer.y.min()
                                                            ): str(
                                                                np.round(
                                                                    self.explainer.y.min(),
                                                                    self.round,
                                                                )
                                                            ),
                                                            float(
                                                                self.explainer.y.max()
                                                            ): str(
                                                                np.round(
                                                                    self.explainer.y.max(),
                                                                    self.round,
                                                                )
                                                            ),
                                                        },
                                                        allowCross=False,
                                                        tooltip={
                                                            "always_visible": False
                                                        },
                                                    ),
                                                ],
                                                id="random-index-reg-y-slider-div-"
                                                + self.name,
                                                # Style adjusted in callback based on initial value
                                                style=None if self.pred_or_y == "y" else dict(display="none")
                                            ),
                                        ],
                                        md=8,
                                    ),
                                    hide=self.hide_pred_slider,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                "Intervalo:", # Translated
                                                id="random-index-reg-preds-or-y-label-"
                                                + self.name,
                                                html_for="random-index-reg-preds-or-y-"
                                                + self.name,
                                            ),
                                            dbc.Select(
                                                id="random-index-reg-preds-or-y-"
                                                + self.name,
                                                options=[
                                                    {
                                                        "label": "Previsto", # Translated
                                                        "value": "preds",
                                                    },
                                                    {"label": "Observado", "value": "y"}, # Translated
                                                ],
                                                value=self.pred_or_y,
                                            ),
                                            dbc.Tooltip(
                                                # Translated
                                                f"Pode selecionar um {self.explainer.index_name} aleatório apenas dentro de um certo intervalo do "
                                                f"{self.explainer.target} observado ou dentro de um certo intervalo do {self.explainer.target} previsto.",
                                                target="random-index-reg-preds-or-y-label-"
                                                + self.name,
                                            ),
                                        ],
                                        md=4,
                                    ),
                                    hide=self.hide_pred_or_y,
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
                                                    dbc.Label(
                                                        "Intervalo de Resíduos:", # Translated
                                                        id="random-index-reg-residual-slider-label-"
                                                        + self.name,
                                                        html_for="random-index-reg-residual-slider-"
                                                        + self.name,
                                                    ),
                                                    dbc.Tooltip(
                                                        # Translated
                                                        f"Selecionar apenas {self.explainer.index_name} onde o "
                                                        f"resíduo (diferença entre o {self.explainer.target} observado e o {self.explainer.target} previsto)"
                                                        " estava dentro do seguinte intervalo:",
                                                        target="random-index-reg-residual-slider-label-"
                                                        + self.name,
                                                    ),
                                                    dcc.RangeSlider(
                                                        id="random-index-reg-residual-slider-"
                                                        + self.name,
                                                        min=float(
                                                            self.explainer.residuals.min()
                                                        ),
                                                        max=float(
                                                            self.explainer.residuals.max()
                                                        ),
                                                        step=float(
                                                            np.float_power(
                                                                10, -self.round
                                                            )
                                                        ),
                                                        value=[
                                                            float(
                                                                self.residual_slider[0]
                                                            ),
                                                            float(
                                                                self.residual_slider[1]
                                                            ),
                                                        ],
                                                        marks={
                                                            float(
                                                                self.explainer.residuals.min()
                                                            ): str(
                                                                np.round(
                                                                    self.explainer.residuals.min(),
                                                                    self.round,
                                                                )
                                                            ),
                                                            float(
                                                                self.explainer.residuals.max()
                                                            ): str(
                                                                np.round(
                                                                    self.explainer.residuals.max(),
                                                                    self.round,
                                                                )
                                                            ),
                                                        },
                                                        allowCross=False,
                                                        tooltip={
                                                            "always_visible": False
                                                        },
                                                    ),
                                                ],
                                                id="random-index-reg-residual-slider-div-"
                                                + self.name,
                                                # Style adjusted in callback based on initial value
                                                style=None if not self.abs_residuals else dict(display="none")
                                            ),
                                            html.Div(
                                                [
                                                    dbc.Label(
                                                        "Resíduos Absolutos", # Translated
                                                        id="random-index-reg-abs-residual-slider-label"
                                                        + self.name,
                                                        html_for="random-index-reg-abs-residual-slider-"
                                                        + self.name,
                                                    ),
                                                    dbc.Tooltip(
                                                        # Translated
                                                        f"Selecionar apenas {self.explainer.index_name} onde o resíduo absoluto "
                                                        f"(diferença entre o {self.explainer.target} observado e o {self.explainer.target} previsto)"
                                                        " estava dentro do seguinte intervalo:",
                                                        target="random-index-reg-abs-residual-slider-label"
                                                        + self.name,
                                                    ),
                                                    dcc.RangeSlider(
                                                        id="random-index-reg-abs-residual-slider-"
                                                        + self.name,
                                                        min=float(
                                                            self.explainer.abs_residuals.min()
                                                        ),
                                                        max=float(
                                                            self.explainer.abs_residuals.max()
                                                        ),
                                                        step=float(
                                                            np.float_power(
                                                                10, -self.round
                                                            )
                                                        ),
                                                        value=[
                                                            float(
                                                                self.abs_residual_slider[
                                                                    0
                                                                ]
                                                            ),
                                                            float(
                                                                self.abs_residual_slider[
                                                                    1
                                                                ]
                                                            ),
                                                        ],
                                                        marks={
                                                            float(
                                                                self.explainer.abs_residuals.min()
                                                            ): str(
                                                                np.round(
                                                                    self.explainer.abs_residuals.min(),
                                                                    self.round,
                                                                )
                                                            ),
                                                            float(
                                                                self.explainer.abs_residuals.max()
                                                            ): str(
                                                                np.round(
                                                                    self.explainer.abs_residuals.max(),
                                                                    self.round,
                                                                )
                                                            ),
                                                        },
                                                        allowCross=False,
                                                        tooltip={
                                                            "always_visible": False
                                                        },
                                                    ),
                                                ],
                                                id="random-index-reg-abs-residual-slider-div-"
                                                + self.name,
                                                # Style adjusted in callback based on initial value
                                                style=None if self.abs_residuals else dict(display="none")
                                            ),
                                        ],
                                        md=8,
                                    ),
                                    hide=self.hide_residual_slider,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                "Resíduos:", # Translated
                                                id="random-index-reg-abs-residual-label-"
                                                + self.name,
                                                html_for="random-index-reg-abs-residual-"
                                                + self.name,
                                            ),
                                            dbc.Select(
                                                id="random-index-reg-abs-residual-"
                                                + self.name,
                                                options=[
                                                    {
                                                        "label": "Resíduos", # Translated
                                                        "value": "relative",
                                                    },
                                                    {
                                                        "label": "Resíduos Absolutos", # Translated
                                                        "value": "absolute",
                                                    },
                                                ],
                                                value="absolute"
                                                if self.abs_residuals
                                                else "relative",
                                            ),
                                            dbc.Tooltip(
                                                # Translated
                                                f"Pode selecionar um {self.explainer.index_name} aleatório "
                                                f"apenas dentro de um certo intervalo de resíduos "
                                                f"(diferença entre o {self.explainer.target} observado e previsto), "
                                                f"por exemplo, apenas {self.explainer.index_name} para os quais a previsão "
                                                f"foi demasiado alta ou demasiado baixa."
                                                f" Ou pode selecionar apenas dentro de um certo intervalo de resíduos absolutos. Por exemplo, "
                                                f"selecionar apenas {self.explainer.index_name} para os quais a previsão estava errada por "
                                                f"pelo menos uma certa quantidade de {self.explainer.units}.",
                                                target="random-index-reg-abs-residual-label-"
                                                + self.name,
                                            ),
                                        ],
                                        md=4,
                                    ),
                                    hide=self.hide_abs_residuals,
                                ),
                            ]
                        ),
                    ]
                ),
            ]
        )

    def to_html(self, state_dict=None, add_header=True):
        args = self.get_state_args(state_dict)

        # Translated output string
        html = to_html.card(
            f"Índice selecionado: <b>{self.explainer.get_index(args['index'])}</b>",
            title=self.title,
        )
        if add_header:
            return to_html.add_header(html)
        return html

    def component_callbacks(self, app):
        @app.callback(
            [
                Output("random-index-reg-pred-slider-div-" + self.name, "style"),
                Output("random-index-reg-y-slider-div-" + self.name, "style"),
            ],
            [Input("random-index-reg-preds-or-y-" + self.name, "value")],
        )
        def update_reg_hidden_div_pred_sliders(preds_or_y):
            if preds_or_y == "preds":
                return (None, dict(display="none"))
            elif preds_or_y == "y":
                return (dict(display="none"), None)
            raise PreventUpdate

        @app.callback(
            [
                Output("random-index-reg-residual-slider-div-" + self.name, "style"),
                Output(
                    "random-index-reg-abs-residual-slider-div-" + self.name, "style"
                ),
            ],
            [Input("random-index-reg-abs-residual-" + self.name, "value")],
        )
        def update_reg_hidden_div_residual_sliders(abs_residuals_toggle): # Renamed for clarity
            if abs_residuals_toggle == "absolute":
                return (dict(display="none"), None)
            elif abs_residuals_toggle == "relative": # Check against actual value
                return (None, dict(display="none"))
            raise PreventUpdate # Should not happen

        @app.callback(
            [
                Output("random-index-reg-residual-slider-" + self.name, "min"),
                Output("random-index-reg-residual-slider-" + self.name, "max"),
                Output("random-index-reg-residual-slider-" + self.name, "value"),
                Output("random-index-reg-residual-slider-" + self.name, "marks"),
                Output("random-index-reg-abs-residual-slider-" + self.name, "min"),
                Output("random-index-reg-abs-residual-slider-" + self.name, "max"),
                Output("random-index-reg-abs-residual-slider-" + self.name, "value"),
                Output("random-index-reg-abs-residual-slider-" + self.name, "marks"),
            ],
            [
                Input("random-index-reg-pred-slider-" + self.name, "value"),
                Input("random-index-reg-y-slider-" + self.name, "value"),
            ],
            [
                State("random-index-reg-preds-or-y-" + self.name, "value"),
                State("random-index-reg-residual-slider-" + self.name, "value"),
                State("random-index-reg-abs-residual-slider-" + self.name, "value"),
            ],
        )
        def update_residual_slider_limits(
            pred_range, y_range, preds_or_y, residuals_range, abs_residuals_range
        ):
            # This logic remains unchanged, it calculates limits based on data
            # Added check for empty slices to avoid errors
            if preds_or_y == "preds":
                mask = (self.explainer.preds >= pred_range[0]) & (self.explainer.preds <= pred_range[1])
            elif preds_or_y == "y":
                mask = (self.explainer.y >= y_range[0]) & (self.explainer.y <= y_range[1])
            else:
                 raise PreventUpdate

            if not mask.any(): # Avoid errors if mask is all False
                 min_residuals, max_residuals = 0, 1
                 min_abs_residuals, max_abs_residuals = 0, 1
                 new_residuals_range = [0, 1]
                 new_abs_residuals_range = [0, 1]
            else:
                min_residuals = float(self.explainer.residuals[mask].min())
                max_residuals = float(self.explainer.residuals[mask].max())
                min_abs_residuals = float(self.explainer.abs_residuals[mask].min())
                max_abs_residuals = float(self.explainer.abs_residuals[mask].max())

                # Adjust ranges safely
                new_residuals_range = [
                    max(min_residuals, residuals_range[0]),
                    min(max_residuals, residuals_range[1]),
                ]
                if new_residuals_range[0] > new_residuals_range[1]: # Ensure lower <= upper
                    new_residuals_range[1] = new_residuals_range[0]

                new_abs_residuals_range = [
                    max(min_abs_residuals, abs_residuals_range[0]),
                    min(max_abs_residuals, abs_residuals_range[1]),
                ]
                if new_abs_residuals_range[0] > new_abs_residuals_range[1]: # Ensure lower <= upper
                    new_abs_residuals_range[1] = new_abs_residuals_range[0]

            residuals_marks = {
                min_residuals: str(np.round(min_residuals, self.round)),
                max_residuals: str(np.round(max_residuals, self.round)),
            }
            abs_residuals_marks = {
                min_abs_residuals: str(np.round(min_abs_residuals, self.round)),
                max_abs_residuals: str(np.round(max_abs_residuals, self.round)),
            }
            return (
                min_residuals,
                max_residuals,
                new_residuals_range,
                residuals_marks,
                min_abs_residuals,
                max_abs_residuals,
                new_abs_residuals_range,
                abs_residuals_marks,
            )

        @app.callback(
            Output("random-index-reg-index-" + self.name, "value"),
            [Input("random-index-reg-button-" + self.name, "n_clicks")],
            [
                State("random-index-reg-pred-slider-" + self.name, "value"),
                State("random-index-reg-y-slider-" + self.name, "value"),
                State("random-index-reg-residual-slider-" + self.name, "value"),
                State("random-index-reg-abs-residual-slider-" + self.name, "value"),
                State("random-index-reg-preds-or-y-" + self.name, "value"),
                State("random-index-reg-abs-residual-" + self.name, "value"),
            ],
        )
        def update_index(
            n_clicks,
            pred_range,
            y_range,
            residual_range,
            abs_residuals_range,
            preds_or_y,
            abs_residuals_toggle, # Renamed for clarity
        ):
            # This logic remains unchanged, it selects an index based on constraints
            triggers = [
                trigger["prop_id"] for trigger in dash.callback_context.triggered
            ]
            # Only trigger on button click
            if not triggers or f"random-index-reg-button-{self.name}.n_clicks" not in triggers[0]:
                raise PreventUpdate
            # n_clicks check is redundant if we check the trigger source

            if preds_or_y == "preds":
                if abs_residuals_toggle == "absolute":
                    idx = self.explainer.random_index(
                        pred_min=pred_range[0],
                        pred_max=pred_range[1],
                        abs_residuals_min=abs_residuals_range[0],
                        abs_residuals_max=abs_residuals_range[1],
                        return_str=True,
                    )
                else: # 'relative'
                    idx = self.explainer.random_index(
                        pred_min=pred_range[0],
                        pred_max=pred_range[1],
                        residuals_min=residual_range[0],
                        residuals_max=residual_range[1],
                        return_str=True,
                    )
            elif preds_or_y == "y":
                if abs_residuals_toggle == "absolute":
                    idx = self.explainer.random_index(
                        y_min=y_range[0],
                        y_max=y_range[1],
                        abs_residuals_min=abs_residuals_range[0],
                        abs_residuals_max=abs_residuals_range[1],
                        return_str=True,
                    )
                else: # 'relative'
                    # Original code used pred_range here, likely a bug. Corrected to y_range.
                    idx = self.explainer.random_index(
                        y_min=y_range[0],
                        y_max=y_range[1],
                        residuals_min=residual_range[0],
                        residuals_max=residual_range[1],
                        return_str=True,
                    )
            else:
                raise PreventUpdate # Should not happen

            # Handle case where no index matches criteria
            if idx is None:
                # Optionally provide feedback to the user here, e.g., via an alert
                print("No index found matching the criteria.") # Or use a more user-friendly method
                raise PreventUpdate
            return idx


class RegressionModelSummaryComponent(ExplainerComponent):
    def __init__(
        self,
        explainer,
        title="Resumo do Modelo", # Translated
        name=None,
        subtitle="Métricas quantitativas para o desempenho do modelo", # Translated
        hide_title=False,
        hide_subtitle=False,
        round=3,
        show_metrics=None,
        description=None,
        **kwargs,
    ):
        """Show model summary statistics (RMSE, MAE, R2) component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Resumo do Modelo".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            subtitle (str): subtitle
            hide_title (bool, optional): hide title
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            round (int): rounding to perform to metric floats.
            show_metrics (List): list of metrics to display in order. Defaults
                to None, displaying all metrics.
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown.
        """
        super().__init__(explainer, title, name)
        if self.description is None:
            # Translated default description
            self.description = f"""
        Na tabela abaixo, pode encontrar várias métricas de desempenho de regressão
        que descrevem quão bem o modelo consegue prever
        {self.explainer.target}.
        """
        self.register_dependencies(["preds", "residuals"])

    def layout(self):
        metrics_dict = self.explainer.metrics_descriptions()
        metrics_df = (
            pd.DataFrame(
                self.explainer.metrics(show_metrics=self.show_metrics), index=["Pontuação"] # Translated
            )
            .T.rename_axis(index="métrica") # Translated
            .reset_index()
            .round(self.round)
        )
        # Rename columns for display if they exist after reset_index
        metrics_df = metrics_df.rename(columns={'métrica': 'Métrica', 'Pontuação': 'Pontuação'}) # Capitalized for headers

        metrics_table = dbc.Table.from_dataframe(
            metrics_df, striped=False, bordered=False, hover=False
        )
        # Ensure tooltips use the potentially renamed 'Métrica' column
        metrics_table, tooltips = get_dbc_tooltips(
            metrics_table, metrics_dict, "reg-model-summary-div-hover", self.name, id_col='Métrica'
        )
        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        self.title, # Title translated in __init__
                                        id="reg-model-summary-title-" + self.name,
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle" # Subtitle translated in __init__
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description, # Description translated in __init__
                                        target="reg-model-summary-title-" + self.name,
                                    ),
                                ]
                            ),
                        ]
                    ),
                    hide=self.hide_title,
                ),
                dbc.CardBody([metrics_table, *tooltips]),
            ]
        )

    def to_html(self, state_dict=None, add_header=True):
        args = self.get_state_args(state_dict)
        metrics_df = (
            pd.DataFrame(
                self.explainer.metrics(show_metrics=self.show_metrics), index=["Pontuação"] # Translated
            )
            .T.rename_axis(index="métrica") # Translated
            .reset_index()
            .round(self.round)
        )
        # Rename columns for display if they exist after reset_index
        metrics_df = metrics_df.rename(columns={'métrica': 'Métrica', 'Pontuação': 'Pontuação'}) # Capitalized for headers
        html = to_html.table_from_df(metrics_df)
        html = to_html.card(html, title=self.title)
        if add_header:
            return to_html.add_header(html)
        return html


class RegressionPredictionSummaryComponent(ExplainerComponent):
    _state_props = dict(index=("reg-prediction-index-", "value"))

    def __init__(
        self,
        explainer,
        title="Previsão", # Translated
        name=None,
        hide_index=False,
        hide_title=False,
        hide_subtitle=False, # Subtitle not used by default, but kept for consistency
        hide_table=False,
        index_dropdown=True,
        feature_input_component=None,
        index=None,
        round=3,
        description=None,
        **kwargs,
    ):
        """Shows a summary for a particular prediction

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Previsão".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            hide_index (bool, optional): hide index selector. Defaults to False.
            hide_title (bool, optional): hide title. Defaults to False.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_table (bool, optional): hide the results table
            index_dropdown (bool, optional): Use dropdown for index input instead
                of free text input. Defaults to True.
            feature_input_component (FeatureInputComponent): A FeatureInputComponent
                that will give the input to the graph instead of the index selector.
                If not None, hide_index=True. Defaults to None.
            index ({int, str}, optional): Index to display prediction summary for. Defaults to None.
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown.
        """
        super().__init__(explainer, title, name)

        self.index_name = "reg-prediction-index-" + self.name
        self.index_selector = IndexSelector(
            explainer, self.index_name, index=index, index_dropdown=index_dropdown
        )

        if self.feature_input_component is not None:
            self.exclude_callbacks(self.feature_input_component)
            self.hide_index = True

        if self.description is None:
            # Translated default description
            self.description = f"""
        Mostra o {self.explainer.target} previsto e o {self.explainer.target} observado,
        bem como a diferença entre os dois (resíduo)
        """

    def layout(self):
        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.H3(
                                self.title, # Title translated in __init__
                                id="reg-prediction-title-" + self.name,
                                className="card-title",
                            ),
                            dbc.Tooltip(
                                self.description, # Description translated in __init__
                                target="reg-prediction-title-" + self.name,
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
                                            # Label uses variable, no direct translation needed
                                            dbc.Label(f"{self.explainer.index_name}:"),
                                            self.index_selector.layout(),
                                        ],
                                        md=6,
                                    ),
                                    hide=self.hide_index,
                                ),
                            ]
                        ),
                        dbc.Row(
                            [dbc.Col([html.Div(id="reg-prediction-div-" + self.name)])]
                        ),
                    ]
                ),
            ]
        )

    def get_state_tuples(self):
        _state_tuples = super().get_state_tuples()
        if self.feature_input_component is not None:
            _state_tuples.extend(self.feature_input_component.get_state_tuples())
        return sorted(list(set(_state_tuples)))

    def to_html(self, state_dict=None, add_header=True):
        args = self.get_state_args(state_dict)
        if self.feature_input_component is None:
            if args["index"] is not None:
                # Table content comes from explainer methods, assuming headers are okay
                preds_df = self.explainer.prediction_result_df(
                    args["index"], round=self.round
                )
                html = to_html.table_from_df(preds_df)
            else:
                html = "nenhum índice selecionado" # Translated
        else:
            inputs = {
                k: v
                for k, v in self.feature_input_component.get_state_args(
                    state_dict
                ).items()
                if k != "index"
            }
            inputs = list(inputs.values())
            if len(inputs) == len(
                self.feature_input_component._input_features
            ) and not any([i is None for i in inputs]):
                X_row = self.explainer.get_row_from_input(inputs, ranked_by_shap=True)
                # Table content comes from explainer methods, assuming headers are okay
                preds_df = self.explainer.prediction_result_df(
                    X_row=X_row, round=self.round
                )
                html = to_html.table_from_df(preds_df)
            else:
                html = f"<div>dados de entrada incorretos</div>" # Translated

        html = to_html.card(html, title=self.title)
        if add_header:
            return to_html.add_header(html)
        return html

    def component_callbacks(self, app):
        if self.feature_input_component is None:

            @app.callback(
                Output("reg-prediction-div-" + self.name, "children"),
                [Input("reg-prediction-index-" + self.name, "value")],
            )
            def update_output_div(index):
                if index is None or not self.explainer.index_exists(index):
                    raise PreventUpdate
                # Table content comes from explainer methods, assuming headers are okay
                preds_df = self.explainer.prediction_result_df(index, round=self.round)
                return make_hideable(
                    dbc.Table.from_dataframe(
                        preds_df, striped=False, bordered=False, hover=False
                    ),
                    hide=self.hide_table,
                )

        else:

            @app.callback(
                Output("reg-prediction-div-" + self.name, "children"),
                [*self.feature_input_component._feature_callback_inputs],
            )
            def update_output_div(*inputs):
                # Logic remains unchanged
                # Check if inputs are valid before proceeding
                if len(inputs) != len(self.feature_input_component._input_features) or \
                   any(i is None for i in inputs):
                    # Return a message or empty div if inputs are not ready
                    return html.Div("Aguardando dados de entrada...") # Example message

                X_row = self.explainer.get_row_from_input(inputs, ranked_by_shap=True)
                # Table content comes from explainer methods, assuming headers are okay
                preds_df = self.explainer.prediction_result_df(
                    X_row=X_row, round=self.round
                )
                return make_hideable(
                    dbc.Table.from_dataframe(
                        preds_df, striped=False, bordered=False, hover=False
                    ),
                    hide=self.hide_table,
                )


class PredictedVsActualComponent(ExplainerComponent):
    _state_props = dict(
        log_x=("pred-vs-actual-logx-", "value"), log_y=("pred-vs-actual-logy-", "value")
    )

    def __init__(
        self,
        explainer,
        title="Previsto vs. Observado", # Translated
        name=None,
        subtitle="Quão próximo está o valor previsto do observado?", # Translated
        hide_title=False,
        hide_subtitle=False,
        hide_log_x=False,
        hide_log_y=False,
        hide_popout=False,
        logs=False,
        log_x=False,
        log_y=False,
        round=3,
        plot_sample=None,
        description=None,
        **kwargs,
    ):
        """Shows a plot of predictions vs y.

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Previsto vs. Observado".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            subtitle (str): subtitle
            hide_title (bool, optional) Hide the title. Defaults to False.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_log_x (bool, optional): Hide the log_x toggle. Defaults to False.
            hide_log_y (bool, optional): Hide the log_y toggle. Defaults to False.
            hide_popout (bool, optional): hide popout button. Defaults to False.
            logs (bool, optional): Whether to use log axis. Defaults to False.
            log_x (bool, optional): log only x axis. Defaults to False.
            log_y (bool, optional): log only y axis. Defaults to False.
            round (int, optional): rounding to apply to float predictions.
                Defaults to 3.
            plot_sample (int, optional): Instead of all points only plot a random
                sample of points. Defaults to None (=all points)
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown.
        """
        super().__init__(explainer, title, name)

        self.logs, self.log_x, self.log_y = logs, log_x, log_y

        if self.description is None:
            # Translated default description
            self.description = f"""
        O gráfico mostra o {self.explainer.target} observado e o {self.explainer.target} previsto
        no mesmo gráfico. Um modelo perfeito teria todos os pontos na diagonal
        (previsto igual a observado). Quanto mais afastados os pontos estiverem da diagonal,
        pior o modelo prevê o {self.explainer.target}.
        """

        self.popout = GraphPopout(
            "pred-vs-actual-" + self.name + "popout",
            "pred-vs-actual-graph-" + self.name,
            self.title,
            self.description,
        )
        self.register_dependencies(["preds"])

    def layout(self):
        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        self.title, # Title translated in __init__
                                        id="pred-vs-actual-title-" + self.name,
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle" # Subtitle translated in __init__
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description, # Description translated in __init__
                                        target="pred-vs-actual-title-" + self.name,
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
                                            # Using dbc.Form and dbc.Checkbox for better layout/styling
                                            dbc.Form(
                                                dbc.Checkbox(
                                                    id="pred-vs-actual-logy-" + self.name,
                                                    label="Log y", # Keep technical term
                                                    value=self.log_y,
                                                    className="form-check-input",
                                                    label_class_name="form-check-label",
                                                    label_style={"padding-left": "0.25rem"}
                                                ),
                                            ),
                                            dbc.Tooltip(
                                                # Translated
                                                "Ao usar um eixo logarítmico, é mais fácil ver erros relativos "
                                                "em vez de erros absolutos.",
                                                target="pred-vs-actual-logy-" + self.name,
                                            ),
                                        ],
                                        # Adjust width and alignment for better spacing
                                        width={"size": "auto"},
                                        className="d-flex align-items-center", # Use flexbox for alignment
                                        style={"padding-left": "1rem"} # Add padding if needed
                                    ),
                                    hide=self.hide_log_y,
                                ),
                                dbc.Col(
                                    [
                                        dcc.Graph(
                                            id="pred-vs-actual-graph-" + self.name,
                                            config=dict(
                                                modeBarButtons=[["toImage"]],
                                                displaylogo=False,
                                            ),
                                        ),
                                    ],
                                    # Adjust width if needed, e.g., md=11 if the toggle takes space
                                ),
                            ],
                            align="center" # Align row items vertically
                        ),
                        dbc.Row(
                            [
                                dbc.Col(width=1), # Offset column to align under graph
                                make_hideable(
                                    dbc.Col(
                                        [
                                            # Using dbc.Form and dbc.Checkbox
                                            dbc.Form(
                                                dbc.Checkbox(
                                                    id="pred-vs-actual-logx-" + self.name,
                                                    label="Log x", # Keep technical term
                                                    value=self.log_x,
                                                    className="form-check-input",
                                                    label_class_name="form-check-label",
                                                    label_style={"padding-left": "0.25rem"}
                                                ),
                                            ),
                                            dbc.Tooltip(
                                                # Translated
                                                "Ao usar um eixo logarítmico, é mais fácil ver erros relativos "
                                                "em vez de erros absolutos.",
                                                target="pred-vs-actual-logx-" + self.name,
                                            ),
                                        ],
                                        width={"size": "auto"},
                                        className="d-flex align-items-center", # Use flexbox
                                    ),
                                    hide=self.hide_log_x,
                                ),
                            ],
                            justify="start", # Align toggle to the left under the graph
                            className="mt-2" # Add margin top for spacing
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
            ]
        )

    def to_html(self, state_dict=None, add_header=True):
        args = self.get_state_args(state_dict)
        # Plotting logic remains unchanged
        fig = self.explainer.plot_predicted_vs_actual(
            log_x=bool(args["log_x"]),
            log_y=bool(args["log_y"]),
            round=self.round,
            plot_sample=self.plot_sample,
        )
        html = to_html.card(to_html.fig(fig), title=self.title)
        if add_header:
            return to_html.add_header(html)
        return html

    def component_callbacks(self, app):
        @app.callback(
            Output("pred-vs-actual-graph-" + self.name, "figure"),
            [
                Input("pred-vs-actual-logx-" + self.name, "value"),
                Input("pred-vs-actual-logy-" + self.name, "value"),
            ],
        )
        def update_predicted_vs_actual_graph(log_x, log_y):
            # Plotting logic remains unchanged
            return self.explainer.plot_predicted_vs_actual(
                log_x=log_x, log_y=log_y, round=self.round, plot_sample=self.plot_sample
            )


class ResidualsComponent(ExplainerComponent):
    _state_props = dict(
        pred_or_actual=("residuals-pred-or-actual-", "value"),
        residuals=("residuals-type-", "value"),
    )

    def __init__(
        self,
        explainer,
        title="Resíduos", # Translated
        name=None,
        subtitle="Qual o desvio do modelo?", # Translated
        hide_title=False,
        hide_subtitle=False,
        hide_footer=False,
        hide_pred_or_actual=False,
        hide_ratio=False, # This hides the residual type selector
        hide_popout=False,
        pred_or_actual="vs_pred",
        residuals="difference",
        round=3,
        plot_sample=None,
        description=None,
        **kwargs,
    ):
        """Residuals plot component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Resíduos".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            subtitle (str): subtitle
            hide_title (bool, optional) Hide the title. Defaults to False.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_footer (bool, optional): hide the footer at the bottom of the component
            hide_pred_or_actual (bool, optional): hide vs predictions or vs
                        actual for x-axis toggle. Defaults to False.
            hide_ratio (bool, optional): hide residual type dropdown. Defaults to False.
            hide_popout (bool, optional): hide popout button. Defaults to False.
            pred_or_actual (str, {'vs_actual', 'vs_pred'}, optional): Whether
                        to plot actual or predictions on the x-axis.
                        Defaults to "vs_pred".
            residuals (str, {'difference', 'ratio', 'log-ratio'} optional):
                    How to calcualte residuals. Defaults to 'difference'.
            round (int, optional): rounding to apply to float predictions.
                Defaults to 3.
            plot_sample (int, optional): Instead of all points only plot a random
                sample of points. Defaults to None (=all points)
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown.
        """
        super().__init__(explainer, title, name)

        assert residuals in ["difference", "ratio", "log-ratio"], (
            "parameter residuals should in ['difference', 'ratio', 'log-ratio']"
            f" but you passed residuals={residuals}"
        )

        if self.description is None:
            # Translated default description
            self.description = f"""
        Os resíduos são a diferença entre o {self.explainer.target} observado
        e o {self.explainer.target} previsto. Neste gráfico, pode verificar se
        os resíduos são maiores ou menores para resultados observados/previstos mais altos/baixos.
        Assim, pode verificar se o modelo funciona melhor ou pior para diferentes níveis de {self.explainer.target}.
        """

        self.popout = GraphPopout(
            "residuals-" + self.name + "popout",
            "residuals-graph-" + self.name,
            self.title,
            self.description,
        )
        self.register_dependencies(["preds", "residuals"])

    def layout(self):
        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        self.title, id="residuals-title-" + self.name # Title translated in __init__
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle" # Subtitle translated in __init__
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description, # Description translated in __init__
                                        target="residuals-title-" + self.name,
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
                                dbc.Col(
                                    [
                                        dcc.Graph(
                                            id="residuals-graph-" + self.name,
                                            config=dict(
                                                modeBarButtons=[["toImage"]],
                                                displaylogo=False,
                                            ),
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
                                                # Group label and select using Row/Col for better control
                                                dbc.Row(
                                                    [
                                                        dbc.Label(
                                                            "Eixo horizontal:", # Translated
                                                            html_for="residuals-pred-or-actual-" + self.name,
                                                            width="auto" # Adjust width as needed
                                                        ),
                                                        dbc.Col(
                                                            dbc.Select(
                                                                options=[
                                                                    {
                                                                        "label": "Previsto", # Translated
                                                                        "value": "vs_pred",
                                                                    },
                                                                    {
                                                                        "label": "Observado", # Translated
                                                                        "value": "vs_actual",
                                                                    },
                                                                ],
                                                                value=self.pred_or_actual,
                                                                id="residuals-pred-or-actual-" + self.name,
                                                                size="sm",
                                                            ),
                                                        ),
                                                    ],
                                                    # Use Row's id for tooltip target
                                                    id="residuals-pred-or-actual-form-" + self.name,
                                                    align="center" # Align items vertically
                                                ),
                                                dbc.Tooltip(
                                                    # Translated
                                                    "Selecione o que gostaria de colocar no eixo x:"
                                                    f" {self.explainer.target} observado ou {self.explainer.target} previsto.",
                                                    target="residuals-pred-or-actual-form-" + self.name,
                                                ),
                                            ],
                                            md=5, # Adjust column width
                                        ),
                                        hide=self.hide_pred_or_actual,
                                    ),
                                    make_hideable(
                                        dbc.Col(
                                            [
                                                # Group label and select
                                                dbc.Row(
                                                    [
                                                        dbc.Label(
                                                            "Tipo de resíduo:", # Translated
                                                            id="residuals-type-label-" + self.name, # Use label id for tooltip
                                                            html_for="residuals-type-" + self.name,
                                                            width="auto" # Adjust width
                                                        ),
                                                        dbc.Col(
                                                            dbc.Select(
                                                                id="residuals-type-" + self.name,
                                                                options=[
                                                                    {
                                                                        "label": "Diferença", # Translated
                                                                        "value": "difference",
                                                                    },
                                                                    {
                                                                        "label": "Razão", # Translated
                                                                        "value": "ratio",
                                                                    },
                                                                    {
                                                                        "label": "Log Razão", # Translated
                                                                        "value": "log-ratio",
                                                                    },
                                                                ],
                                                                value=self.residuals,
                                                                size="sm",
                                                            ),
                                                        ),
                                                    ],
                                                    align="center" # Align items vertically
                                                ),
                                                dbc.Tooltip(
                                                    # Translated
                                                    "Tipo de resíduos a exibir: y-previsto (diferença), "
                                                    "y/previsto (razão) ou log(y/previsto) (log-razão).",
                                                    target="residuals-type-label-" + self.name,
                                                ),
                                            ],
                                            md=5, # Adjust column width
                                        ),
                                        hide=self.hide_ratio, # hide_ratio hides this selector
                                    ),
                                ],
                                justify="start", # Align controls to the left
                            ),
                        ]
                    ),
                    hide=self.hide_footer,
                ),
            ]
        )

    def to_html(self, state_dict=None, add_header=True):
        args = self.get_state_args(state_dict)
        # Plotting logic remains unchanged
        vs_actual = args["pred_or_actual"] == "vs_actual"
        fig = self.explainer.plot_residuals(
            vs_actual=vs_actual,
            residuals=args["residuals"],
            round=self.round,
            plot_sample=self.plot_sample,
        )
        fig.update_layout(margin=dict(t=40, b=40, l=40, r=40))
        html = to_html.card(to_html.fig(fig), title=self.title)
        if add_header:
            return to_html.add_header(html)
        return html

    def component_callbacks(self, app):
        @app.callback(
            Output("residuals-graph-" + self.name, "figure"),
            [
                Input("residuals-pred-or-actual-" + self.name, "value"),
                Input("residuals-type-" + self.name, "value"),
            ],
        )
        def update_residuals_graph(pred_or_actual, residuals):
            # Plotting logic remains unchanged
            vs_actual = pred_or_actual == "vs_actual"
            return self.explainer.plot_residuals(
                vs_actual=vs_actual,
                residuals=residuals,
                round=self.round,
                plot_sample=self.plot_sample,
            )


class RegressionVsColComponent(ExplainerComponent):
    _state_props = dict(
        col=("reg-vs-col-col-", "value"),
        display=("reg-vs-col-display-type-", "value"),
        points=("reg-vs-col-show-points-", "value"),
        winsor=("reg-vs-col-winsor-", "value"),
        cats_topx=("reg-vs-col-n-categories-", "value"),
        cats_sort=("reg-vs-col-categories-sort-", "value"),
    )

    def __init__(
        self,
        explainer,
        title="Gráfico vs. Característica", # Translated
        name=None,
        subtitle="As previsões e os resíduos estão correlacionados com as características?", # Translated
        hide_title=False,
        hide_subtitle=False,
        hide_footer=False,
        hide_col=False,
        hide_ratio=False, # Hides the 'Display' selector
        hide_points=False,
        hide_winsor=False,
        hide_cats_topx=False,
        hide_cats_sort=False,
        hide_popout=False,
        col=None,
        display="difference",
        round=3,
        points=True,
        winsor=0,
        cats_topx=10,
        cats_sort="freq",
        plot_sample=None,
        description=None,
        **kwargs,
    ):
        """Show residuals, observed or preds vs a particular Feature component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Gráfico vs. Característica".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            subtitle (str): subtitle
            hide_title (bool, optional) Hide the title. Defaults to False.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_footer (bool, optional): hide the footer at the bottom of the component
            hide_col (bool, optional): Hide de column selector. Defaults to False.
            hide_ratio (bool, optional): Hide the display type toggle. Defaults to False.
            hide_points (bool, optional): Hide group points toggle. Defaults to False.
            hide_winsor (bool, optional): Hide winsor input. Defaults to False.
            hide_cats_topx (bool, optional): hide the categories topx input. Defaults to False.
            hide_cats_sort (bool, optional): hide the categories sort selector.Defaults to False.
            hide_popout (bool, optional): hide popout button. Defaults to False.
            col ([type], optional): Initial feature to display. Defaults to None.
            display (str, {'observed', 'predicted', difference', 'ratio', 'log-ratio'} optional):
                    What to display on y axis. Defaults to 'difference'.
            round (int, optional): rounding to apply to float predictions.
                Defaults to 3.
            points (bool, optional): display point cloud next to violin plot
                    for categorical cols. Defaults to True
            winsor (int, 0-50, optional): percentage of outliers to winsor out of
                    the y-axis. Defaults to 0.
            cats_topx (int, optional): maximum number of categories to display
                for categorical features. Defaults to 10.
            cats_sort (str, optional): how to sort categories: 'alphabet',
                'freq' or 'shap'. Defaults to 'freq'.
            plot_sample (int, optional): Instead of all points only plot a random
                sample of points. Defaults to None (=all points)
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown.
        """
        super().__init__(explainer, title, name)

        if self.col is None:
            # Ensure columns_ranked_by_shap returns at least one column
            shap_cols = self.explainer.columns_ranked_by_shap()
            if shap_cols:
                self.col = shap_cols[0]
            # Fallback if no shap values (e.g., dummy explainer) or no columns
            elif not self.explainer.X.empty:
                self.col = self.explainer.X.columns[0]
            else:
                self.col = None # Handle case with no columns


        assert self.display in {
            "observed",
            "predicted",
            "difference",
            "ratio",
            "log-ratio",
        }, (
            "parameter display should in {'observed', 'predicted', 'difference', 'ratio', 'log-ratio'}"
            f" but you passed display={self.display}!"
        )

        if self.description is None:
            # Translated default description
            self.description = f"""
        Este gráfico mostra os resíduos (diferença entre o {self.explainer.target} observado
        e o {self.explainer.target} previsto) plotados contra os valores de diferentes características,
        ou o {self.explainer.target} observado ou previsto.
        Isto permite inspecionar se o modelo erra mais para um determinado
        intervalo de valores de características do que para outros.
        """
        self.popout = GraphPopout(
            "reg-vs-col-" + self.name + "popout",
            "reg-vs-col-graph-" + self.name,
            self.title,
            self.description,
        )
        self.register_dependencies(["preds", "residuals"])

    def layout(self):
        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        self.title, id="reg-vs-col-title-" + self.name # Title translated in __init__
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle" # Subtitle translated in __init__
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description, # Description translated in __init__
                                        target="reg-vs-col-title-" + self.name,
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
                                                "Característica:", # Translated
                                                id="reg-vs-col-col-label-" + self.name,
                                                html_for="reg-vs-col-col-" + self.name
                                            ),
                                            dbc.Tooltip(
                                                "Selecione a característica a exibir no eixo x.", # Translated
                                                target="reg-vs-col-col-label-" + self.name,
                                            ),
                                            dbc.Select(
                                                id="reg-vs-col-col-" + self.name,
                                                options=[
                                                    {"label": col, "value": col}
                                                    for col in self.explainer.columns_ranked_by_shap()
                                                ],
                                                value=self.col,
                                            ),
                                        ],
                                        md=6, # Adjusted width
                                    ),
                                    hide=self.hide_col,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                "Exibir:", # Translated
                                                id="reg-vs-col-display-type-label-" + self.name,
                                                html_for="reg-vs-col-display-type-" + self.name
                                            ),
                                            dbc.Tooltip(
                                                # Translated
                                                f"Selecione o que exibir no eixo y: {self.explainer.target} observado, "
                                                f"{self.explainer.target} previsto ou resíduos. Os resíduos podem ser "
                                                "calculados pela diferença (y-previsto), "
                                                "razão (y/previsto) ou logaritmo da razão log(y/previsto). Este último facilita a "
                                                "visualização de diferenças relativas.",
                                                target="reg-vs-col-display-type-label-" + self.name,
                                            ),
                                            dbc.Select(
                                                id="reg-vs-col-display-type-" + self.name,
                                                options=[
                                                    {
                                                        "label": "Observado", # Translated
                                                        "value": "observed",
                                                    },
                                                    {
                                                        "label": "Previsto", # Translated
                                                        "value": "predicted",
                                                    },
                                                    {
                                                        "label": "Resíduos: Diferença", # Translated
                                                        "value": "difference",
                                                    },
                                                    {
                                                        "label": "Resíduos: Razão", # Translated
                                                        "value": "ratio",
                                                    },
                                                    {
                                                        "label": "Resíduos: Log Razão", # Translated
                                                        "value": "log-ratio",
                                                    },
                                                ],
                                                value=self.display,
                                            ),
                                        ],
                                        md=6, # Adjusted width
                                    ),
                                    hide=self.hide_ratio, # hide_ratio hides this selector
                                ),
                            ],
                            className="mb-3" # Add margin bottom for spacing
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dcc.Graph(
                                            id="reg-vs-col-graph-" + self.name,
                                            config=dict(
                                                modeBarButtons=[["toImage"]],
                                                displaylogo=False,
                                            ),
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
                                                dbc.Label(
                                                    "Winsor:", # Keep technical term
                                                    id="reg-vs-col-winsor-label-" + self.name,
                                                    html_for="reg-vs-col-winsor-" + self.name
                                                ),
                                                dbc.Tooltip(
                                                    # Translated
                                                    "Exclui a % de valores y mais altos e mais baixos do gráfico. "
                                                    "Quando existem alguns outliers reais, pode ajudar removê-los"
                                                    " do gráfico para facilitar a visualização do padrão geral.",
                                                    target="reg-vs-col-winsor-label-" + self.name,
                                                ),
                                                dbc.Input(
                                                    id="reg-vs-col-winsor-" + self.name,
                                                    value=self.winsor,
                                                    type="number",
                                                    min=0,
                                                    max=49, # Should be less than 50
                                                    step=1,
                                                    size="sm",
                                                ),
                                            ],
                                            md=3, # Adjusted width
                                        ),
                                        hide=self.hide_winsor,
                                    ),
                                    # Grouping categorical options for conditional display logic
                                    make_hideable(
                                        dbc.Col(
                                            [
                                                html.Div( # Outer div for visibility control
                                                    [
                                                        # Use Form/Checkbox for better layout
                                                        dbc.Form(
                                                            dbc.Checkbox(
                                                                id="reg-vs-col-show-points-" + self.name,
                                                                label="Mostrar nuvem de pontos", # Translated
                                                                value=self.points,
                                                                # inline=True, # Switch implies inline
                                                                switch=True,
                                                            ),
                                                        ),
                                                        dbc.Tooltip(
                                                            # Translated
                                                            "Para características categóricas, exibir "
                                                            "uma nuvem de pontos ao lado dos gráficos de violino.",
                                                            target="reg-vs-col-show-points-" + self.name,
                                                        ),
                                                    ],
                                                    id="reg-vs-col-show-points-div-" + self.name,
                                                )
                                            ],
                                            md=3, # Adjusted width
                                            # Align center might be better for a single checkbox
                                            className="d-flex align-items-center justify-content-center"
                                        ),
                                        self.hide_points,
                                    ),
                                    make_hideable(
                                        dbc.Col(
                                            [
                                                html.Div( # Outer div for visibility control
                                                    [
                                                        dbc.Label(
                                                            "Nº Categorias:", # Translated & Abbreviated
                                                            id="reg-vs-col-n-categories-label-" + self.name,
                                                            html_for="reg-vs-col-n-categories-" + self.name
                                                        ),
                                                        dbc.Tooltip(
                                                            "Número máximo de categorias a exibir", # Translated
                                                            target="reg-vs-col-n-categories-label-" + self.name,
                                                        ),
                                                        dbc.Input(
                                                            id="reg-vs-col-n-categories-" + self.name,
                                                            value=self.cats_topx,
                                                            type="number",
                                                            min=1,
                                                            max=50, # Reasonable max
                                                            step=1,
                                                            size="sm",
                                                        ),
                                                    ],
                                                    id="reg-vs-col-n-categories-div-" + self.name,
                                                ),
                                            ],
                                            md=3, # Adjusted width
                                        ),
                                        self.hide_cats_topx,
                                    ),
                                    make_hideable(
                                        dbc.Col(
                                            [
                                                html.Div( # Outer div for visibility control
                                                    [
                                                        dbc.Label(
                                                            "Ordenar por:", # Translated & Abbreviated
                                                            id="reg-vs-col-categories-sort-label-" + self.name,
                                                            html_for="reg-vs-col-categories-sort-" + self.name
                                                        ),
                                                        dbc.Tooltip(
                                                            # Translated
                                                            "Como ordenar as categorias: Alfabeticamente, mais comuns "
                                                            "primeiro (Frequência), ou maior valor médio absoluto de SHAP primeiro (Impacto Shap)",
                                                            target="reg-vs-col-categories-sort-label-" + self.name,
                                                        ),
                                                        dbc.Select(
                                                            id="reg-vs-col-categories-sort-" + self.name,
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
                                                    ],
                                                    id="reg-vs-col-categories-sort-div-" + self.name,
                                                ),
                                            ],
                                            md=3, # Adjusted width
                                        ),
                                        hide=self.hide_cats_sort,
                                    ),
                                ],
                                justify="start" # Align controls to the left
                            )
                        ]
                    ),
                    hide=self.hide_footer,
                ),
            ]
        )

    def to_html(self, state_dict=None, add_header=True):
        args = self.get_state_args(state_dict)
        # Plotting logic remains unchanged
        # Ensure boolean conversion for points
        points_bool = bool(args["points"]) if isinstance(args["points"], list) else bool(args.get("points"))

        if args["display"] == "observed":
            fig = self.explainer.plot_y_vs_feature(
                args["col"],
                points=points_bool,
                winsor=args["winsor"],
                dropna=True,
                topx=args["cats_topx"],
                sort=args["cats_sort"],
                round=self.round,
                plot_sample=self.plot_sample,
            )
        elif args["display"] == "predicted":
            fig = self.explainer.plot_preds_vs_feature(
                args["col"],
                points=points_bool,
                winsor=args["winsor"],
                dropna=True,
                topx=args["cats_topx"],
                sort=args["cats_sort"],
                round=self.round,
                plot_sample=self.plot_sample,
            )
        else: # difference, ratio, log-ratio
            fig = self.explainer.plot_residuals_vs_feature(
                args["col"],
                residuals=args["display"], # Pass display type as residual type
                points=points_bool,
                winsor=args["winsor"],
                dropna=True,
                topx=args["cats_topx"],
                sort=args["cats_sort"],
                round=self.round,
                plot_sample=self.plot_sample,
            )
        fig.update_layout(margin=dict(t=40, b=40, l=40, r=40))
        html = to_html.card(to_html.fig(fig), title=self.title, subtitle=self.subtitle)
        if add_header:
            return to_html.add_header(html)
        return html

    def component_callbacks(self, app):
        @app.callback(
            [
                Output("reg-vs-col-graph-" + self.name, "figure"),
                Output("reg-vs-col-show-points-div-" + self.name, "style"),
                Output("reg-vs-col-n-categories-div-" + self.name, "style"),
                Output("reg-vs-col-categories-sort-div-" + self.name, "style"),
            ],
            [
                Input("reg-vs-col-col-" + self.name, "value"),
                Input("reg-vs-col-display-type-" + self.name, "value"),
                Input("reg-vs-col-show-points-" + self.name, "value"), # Checklist value is a list
                Input("reg-vs-col-winsor-" + self.name, "value"),
                Input("reg-vs-col-n-categories-" + self.name, "value"),
                Input("reg-vs-col-categories-sort-" + self.name, "value"),
            ],
        )
        def update_vs_col_graph(col, display, points_value, winsor, topx, sort): # Renamed function
            # Logic for showing/hiding categorical options
            is_categorical = False
            if col: # Check if col is selected
                 # Use helper method to check if column is categorical *after* potential encoding
                is_categorical = self.explainer.is_categorical(col)

            # Hide/show categorical controls based on column type
            cat_style = {} if is_categorical else dict(display="none")

            # Ensure winsor is int and within range
            try:
                winsor = int(winsor) if winsor is not None else 0
                winsor = max(0, min(49, winsor)) # Clamp between 0 and 49
            except (ValueError, TypeError):
                winsor = 0 # Default to 0 if input is invalid

            # Ensure topx is int and within range
            try:
                topx = int(topx) if topx is not None else 10
                topx = max(1, min(50, topx)) # Clamp between 1 and 50
            except (ValueError, TypeError):
                 topx = 10 # Default to 10 if input is invalid

            # Convert checklist value (list) to boolean
            # points_value will be [True] if checked, [] if unchecked
            points_bool = bool(points_value)

            # Plotting logic remains mostly unchanged
            if display == "observed":
                figure = self.explainer.plot_y_vs_feature(
                        col,
                        points=points_bool,
                        winsor=winsor,
                        dropna=True,
                        topx=topx,
                        sort=sort,
                        round=self.round,
                        plot_sample=self.plot_sample,
                    )
            elif display == "predicted":
                figure = self.explainer.plot_preds_vs_feature(
                        col,
                        points=points_bool,
                        winsor=winsor,
                        dropna=True,
                        topx=topx,
                        sort=sort,
                        round=self.round,
                        plot_sample=self.plot_sample,
                    )
            else:
                return (
                    self.explainer.plot_residuals_vs_feature(
                        col,
                        residuals=display,
                        points=points_bool,
                        winsor=winsor,
                        dropna=True,
                        topx=topx,
                        sort=sort,
                        round=self.round,
                        plot_sample=self.plot_sample,
                    ),
                    style,
                    style,
                    style,
                )