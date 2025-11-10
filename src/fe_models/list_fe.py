from src.fe_models.autofeat_fe import AutoFeatTransformer 
from src.fe_models.polynomial_fe import PolynomialFeatTransformer
from src.fe_models.featuretools_fe import FeatureToolsTransformer
from src.fe_models.boruta_fe import BorutaFeatureSelector



def get_list_available_fes():
    """
    Define y retorna la lista de transformadores de Feature Engineering disponibles
    para el proceso de encadenamiento.
    """
    available_fes = [
        ("Autofeat", AutoFeatTransformer()),
        ("Polynomial d=2", PolynomialFeatTransformer(degree=2)),
        ("Polynomial d=4", PolynomialFeatTransformer(degree=4)),
        ("FeatureTools (Add/Mult)", FeatureToolsTransformer(max_depth=1)),
        ("Boruta Selector", BorutaFeatureSelector(max_iter=50)),
    ]
    return available_fes
