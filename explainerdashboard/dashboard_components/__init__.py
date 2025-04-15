# explainerdashboard/dashboard_components/__init__.py

# Importações base necessárias por outros módulos
from ..explainers import BaseExplainer, ClassifierExplainer, RegressionExplainer
from ..dashboard_methods import ExplainerComponent, PosLabelSelector, IndexSelector, GraphPopout, make_hideable # Importe outros métodos/classes base se necessário

# Importar os componentes específicos que você QUER que sejam acessíveis via
# from explainerdashboard.dashboard_components import ...
from .overview_components import (
    PredictionSummaryComponent,
    ImportancesComponent,
    FeatureDescriptionsComponent,
    FeatureInputComponent,
    PdpComponent,
)
from .classifier_components import (
    ClassifierRandomIndexComponent,
    ClassifierPredictionSummaryComponent,
    PrecisionComponent,
    ConfusionMatrixComponent,
    LiftCurveComponent,
    ClassificationComponent,
    RocAucComponent,
    PrAucComponent,
    CumulativePrecisionComponent,
    ClassifierModelSummaryComponent,
)
from .regression_components import (
    RegressionRandomIndexComponent,
    RegressionPredictionSummaryComponent,
    PredictedVsActualComponent,
    ResidualsComponent,
    RegressionVsColComponent,
    RegressionModelSummaryComponent,
)
from .shap_components import (
    ShapSummaryComponent,
    ShapDependenceComponent,
    # ShapSummaryDependenceConnector, # Connectors talvez não precisem ser exportados aqui
    InteractionSummaryComponent,
    InteractionDependenceComponent,
    # InteractionSummaryDependenceConnector, # Connectors talvez não precisem ser exportados aqui
    ShapContributionsTableComponent,
    ShapContributionsGraphComponent,
)
from .decisiontree_components import (
    DecisionTreesComponent,
    DecisionPathTableComponent,
    DecisionPathGraphComponent,
)
# from .connectors import * # Pode ou não precisar disto, dependendo do uso externo

# Definir explicitamente o que este __init__ exporta
# Liste TODAS as classes de componentes que você quer que sejam importáveis
# diretamente de dashboard_components
__all__ = [
    # Base/Common Methods (se necessário exportar)
    "ExplainerComponent", "PosLabelSelector", "IndexSelector", "GraphPopout", "make_hideable",

    # Overview Components
    "PredictionSummaryComponent", "ImportancesComponent", "FeatureDescriptionsComponent",
    "FeatureInputComponent", "PdpComponent",

    # Classifier Components
    "ClassifierRandomIndexComponent", "ClassifierPredictionSummaryComponent",
    "PrecisionComponent", "ConfusionMatrixComponent", "LiftCurveComponent",
    "ClassificationComponent", "RocAucComponent", "PrAucComponent",
    "CumulativePrecisionComponent", "ClassifierModelSummaryComponent",

    # Regression Components
    "RegressionRandomIndexComponent", "RegressionPredictionSummaryComponent",
    "PredictedVsActualComponent", "ResidualsComponent", "RegressionVsColComponent",
    "RegressionModelSummaryComponent",

    # SHAP Components
    "ShapSummaryComponent", "ShapDependenceComponent",
    "InteractionSummaryComponent", "InteractionDependenceComponent",
    "ShapContributionsTableComponent", "ShapContributionsGraphComponent",

    # Decision Tree Components
    "DecisionTreesComponent", "DecisionPathTableComponent", "DecisionPathGraphComponent",

    # Adicione quaisquer outros componentes que possam faltar
]
