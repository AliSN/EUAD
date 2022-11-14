from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.ocsvm import OCSVM
from pyod.models.lof import LOF
n_jobs = -1
def base_models():
    randomness_flags = []
    BASE_ESTIMATORS = [
        IForest(n_estimators=10, contamination=0.005, n_jobs = n_jobs),
        IForest(n_estimators=10, contamination=0.01, n_jobs = n_jobs),
        IForest(n_estimators=10, contamination=0.05, n_jobs = n_jobs),

        IForest(n_estimators=20, contamination=0.005, n_jobs = n_jobs),
        IForest(n_estimators=20, contamination=0.01, n_jobs = n_jobs),
        IForest(n_estimators=20, contamination=0.05, n_jobs = n_jobs),
        
        IForest(n_estimators=30, contamination=0.005, n_jobs = n_jobs),
        IForest(n_estimators=30, contamination=0.01, n_jobs = n_jobs),
        IForest(n_estimators=30, contamination=0.05, n_jobs = n_jobs),

        IForest(n_estimators=40, contamination=0.005, n_jobs = n_jobs),
        IForest(n_estimators=40, contamination=0.01, n_jobs = n_jobs),
        IForest(n_estimators=40, contamination=0.05, n_jobs = n_jobs),
        
        IForest(n_estimators=50, contamination=0.005, n_jobs = n_jobs),
        IForest(n_estimators=50, contamination=0.01, n_jobs = n_jobs),
        IForest(n_estimators=50, contamination=0.05, n_jobs = n_jobs),

        IForest(n_estimators=75, contamination=0.005, n_jobs = n_jobs),
        IForest(n_estimators=75, contamination=0.01, n_jobs = n_jobs),
        IForest(n_estimators=75, contamination=0.05, n_jobs = n_jobs),
        
        IForest(n_estimators=100, contamination=0.005, n_jobs = n_jobs),
        IForest(n_estimators=100, contamination=0.01, n_jobs = n_jobs),
        IForest(n_estimators=100, contamination=0.05, n_jobs = n_jobs),

        IForest(n_estimators=150, contamination=0.005, n_jobs = n_jobs),
        IForest(n_estimators=150, contamination=0.01, n_jobs = n_jobs),
        IForest(n_estimators=150, contamination=0.05, n_jobs = n_jobs),
        
        IForest(n_estimators=200, contamination=0.005, n_jobs = n_jobs),
        IForest(n_estimators=200, contamination=0.01, n_jobs = n_jobs),
        IForest(n_estimators=200, contamination=0.05, n_jobs = n_jobs),


        KNN(n_neighbors=1, method='largest', contamination=0.005, n_jobs = n_jobs),
        KNN(n_neighbors=5, method='largest', contamination=0.005, n_jobs = n_jobs),
        KNN(n_neighbors=15, method='largest', contamination=0.005, n_jobs = n_jobs),
        KNN(n_neighbors=25, method='largest', contamination=0.005, n_jobs = n_jobs),
        KNN(n_neighbors=50, method='largest', contamination=0.005, n_jobs = n_jobs),
        KNN(n_neighbors=70, method='largest', contamination=0.005, n_jobs = n_jobs),
        KNN(n_neighbors=100, method='largest', contamination=0.005, n_jobs = n_jobs),
        
        KNN(n_neighbors=1, method='largest', contamination=0.01, n_jobs = n_jobs),
        KNN(n_neighbors=5, method='largest', contamination=0.01, n_jobs = n_jobs),
        KNN(n_neighbors=15, method='largest', contamination=0.01, n_jobs = n_jobs),
        KNN(n_neighbors=25, method='largest', contamination=0.01, n_jobs = n_jobs),
        KNN(n_neighbors=50, method='largest', contamination=0.01, n_jobs = n_jobs),
        KNN(n_neighbors=70, method='largest', contamination=0.01, n_jobs = n_jobs),
        KNN(n_neighbors=100, method='largest', contamination=0.01, n_jobs = n_jobs),
        
        KNN(n_neighbors=1, method='largest', contamination=0.05, n_jobs = n_jobs),
        KNN(n_neighbors=5, method='largest', contamination=0.05, n_jobs = n_jobs),
        KNN(n_neighbors=15, method='largest', contamination=0.05, n_jobs = n_jobs),
        KNN(n_neighbors=25, method='largest', contamination=0.05, n_jobs = n_jobs),
        KNN(n_neighbors=50, method='largest', contamination=0.05, n_jobs = n_jobs),
        KNN(n_neighbors=70, method='largest', contamination=0.05, n_jobs = n_jobs),
        KNN(n_neighbors=100, method='largest', contamination=0.05, n_jobs = n_jobs),
        
        KNN(n_neighbors=1, method='mean', contamination=0.005, n_jobs = n_jobs),
        KNN(n_neighbors=5, method='mean', contamination=0.005, n_jobs = n_jobs),
        KNN(n_neighbors=15, method='mean', contamination=0.005, n_jobs = n_jobs),
        KNN(n_neighbors=25, method='mean', contamination=0.005, n_jobs = n_jobs),
        KNN(n_neighbors=50, method='mean', contamination=0.005, n_jobs = n_jobs),
        KNN(n_neighbors=70, method='mean', contamination=0.005, n_jobs = n_jobs),
        KNN(n_neighbors=100, method='mean', contamination=0.005, n_jobs = n_jobs),
        
        KNN(n_neighbors=1, method='mean', contamination=0.01, n_jobs = n_jobs),
        KNN(n_neighbors=5, method='mean', contamination=0.01, n_jobs = n_jobs),
        KNN(n_neighbors=15, method='mean', contamination=0.01, n_jobs = n_jobs),
        KNN(n_neighbors=25, method='mean', contamination=0.01, n_jobs = n_jobs),
        KNN(n_neighbors=50, method='mean', contamination=0.01, n_jobs = n_jobs),
        KNN(n_neighbors=70, method='mean', contamination=0.01, n_jobs = n_jobs),
        KNN(n_neighbors=100, method='mean', contamination=0.01, n_jobs = n_jobs),
        
        KNN(n_neighbors=1, method='mean', contamination=0.05, n_jobs = n_jobs),
        KNN(n_neighbors=5, method='mean', contamination=0.05, n_jobs = n_jobs),
        KNN(n_neighbors=15, method='mean', contamination=0.05, n_jobs = n_jobs),
        KNN(n_neighbors=25, method='mean', contamination=0.05, n_jobs = n_jobs),
        KNN(n_neighbors=50, method='mean', contamination=0.05, n_jobs = n_jobs),
        KNN(n_neighbors=70, method='mean', contamination=0.05, n_jobs = n_jobs),
        KNN(n_neighbors=100, method='mean', contamination=0.05, n_jobs = n_jobs),

        KNN(n_neighbors=1, method='median', contamination=0.005, n_jobs = n_jobs),
        KNN(n_neighbors=5, method='median', contamination=0.005, n_jobs = n_jobs),
        KNN(n_neighbors=15, method='median', contamination=0.005, n_jobs = n_jobs),
        KNN(n_neighbors=25, method='median', contamination=0.005, n_jobs = n_jobs),
        KNN(n_neighbors=50, method='median', contamination=0.005, n_jobs = n_jobs),
        KNN(n_neighbors=70, method='median', contamination=0.005, n_jobs = n_jobs),
        KNN(n_neighbors=100, method='median', contamination=0.005, n_jobs = n_jobs),
        
        KNN(n_neighbors=1, method='median', contamination=0.01, n_jobs = n_jobs),
        KNN(n_neighbors=5, method='median', contamination=0.01, n_jobs = n_jobs),
        KNN(n_neighbors=15, method='median', contamination=0.01, n_jobs = n_jobs),
        KNN(n_neighbors=25, method='median', contamination=0.01, n_jobs = n_jobs),
        KNN(n_neighbors=50, method='median', contamination=0.01, n_jobs = n_jobs),
        KNN(n_neighbors=70, method='median', contamination=0.01, n_jobs = n_jobs),
        KNN(n_neighbors=100, method='median', contamination=0.01, n_jobs = n_jobs),
        
        KNN(n_neighbors=1, method='median', contamination=0.05, n_jobs = n_jobs),
        KNN(n_neighbors=5, method='median', contamination=0.05, n_jobs = n_jobs),
        KNN(n_neighbors=15, method='median', contamination=0.05, n_jobs = n_jobs),
        KNN(n_neighbors=25, method='median', contamination=0.05, n_jobs = n_jobs),
        KNN(n_neighbors=50, method='median', contamination=0.05, n_jobs = n_jobs),
        KNN(n_neighbors=70, method='median', contamination=0.05, n_jobs = n_jobs),
        KNN(n_neighbors=100, method='median', contamination=0.05, n_jobs = n_jobs),
        

        OCSVM(nu=0.1, kernel="linear", contamination=0.005),
        OCSVM(nu=0.3, kernel="linear", contamination=0.005),
        OCSVM(nu=0.5, kernel="linear", contamination=0.005),
        OCSVM(nu=0.7, kernel="linear", contamination=0.005),
        OCSVM(nu=0.9, kernel="linear", contamination=0.005),
        
        OCSVM(nu=0.1, kernel="linear", contamination=0.01),
        OCSVM(nu=0.3, kernel="linear", contamination=0.01),
        OCSVM(nu=0.5, kernel="linear", contamination=0.01),
        OCSVM(nu=0.7, kernel="linear", contamination=0.01),
        OCSVM(nu=0.9, kernel="linear", contamination=0.01),
        
        OCSVM(nu=0.1, kernel="linear", contamination=0.05),
        OCSVM(nu=0.3, kernel="linear", contamination=0.05),
        OCSVM(nu=0.5, kernel="linear", contamination=0.05),
        OCSVM(nu=0.7, kernel="linear", contamination=0.05),
        OCSVM(nu=0.9, kernel="linear", contamination=0.05),
        
        OCSVM(nu=0.1, kernel="poly", contamination=0.005),
        OCSVM(nu=0.3, kernel="poly", contamination=0.005),
        OCSVM(nu=0.5, kernel="poly", contamination=0.005),
        OCSVM(nu=0.7, kernel="poly", contamination=0.005),
        OCSVM(nu=0.9, kernel="poly", contamination=0.005),
        
        OCSVM(nu=0.1, kernel="poly", contamination=0.01),
        OCSVM(nu=0.3, kernel="poly", contamination=0.01),
        OCSVM(nu=0.5, kernel="poly", contamination=0.01),
        OCSVM(nu=0.7, kernel="poly", contamination=0.01),
        OCSVM(nu=0.9, kernel="poly", contamination=0.01),
        
        OCSVM(nu=0.1, kernel="poly", contamination=0.05),
        OCSVM(nu=0.3, kernel="poly", contamination=0.05),
        OCSVM(nu=0.5, kernel="poly", contamination=0.05),
        OCSVM(nu=0.7, kernel="poly", contamination=0.05),
        OCSVM(nu=0.9, kernel="poly", contamination=0.05),
        
        OCSVM(nu=0.1, kernel="rbf", contamination=0.005),
        OCSVM(nu=0.3, kernel="rbf", contamination=0.005),
        OCSVM(nu=0.5, kernel="rbf", contamination=0.005),
        OCSVM(nu=0.7, kernel="rbf", contamination=0.005),
        OCSVM(nu=0.9, kernel="rbf", contamination=0.005),
        
        OCSVM(nu=0.1, kernel="rbf", contamination=0.01),
        OCSVM(nu=0.3, kernel="rbf", contamination=0.01),
        OCSVM(nu=0.5, kernel="rbf", contamination=0.01),
        OCSVM(nu=0.7, kernel="rbf", contamination=0.01),
        OCSVM(nu=0.9, kernel="rbf", contamination=0.01),
        
        OCSVM(nu=0.1, kernel="rbf", contamination=0.05),
        OCSVM(nu=0.3, kernel="rbf", contamination=0.05),
        OCSVM(nu=0.5, kernel="rbf", contamination=0.05),
        OCSVM(nu=0.7, kernel="rbf", contamination=0.05),
        OCSVM(nu=0.9, kernel="rbf", contamination=0.05),
        
        OCSVM(nu=0.1, kernel="sigmoid", contamination=0.005),
        OCSVM(nu=0.3, kernel="sigmoid", contamination=0.005),
        OCSVM(nu=0.5, kernel="sigmoid", contamination=0.005),
        OCSVM(nu=0.7, kernel="sigmoid", contamination=0.005),
        OCSVM(nu=0.9, kernel="sigmoid", contamination=0.005),
        
        OCSVM(nu=0.1, kernel="sigmoid", contamination=0.01),
        OCSVM(nu=0.3, kernel="sigmoid", contamination=0.01),
        OCSVM(nu=0.5, kernel="sigmoid", contamination=0.01),
        OCSVM(nu=0.7, kernel="sigmoid", contamination=0.01),
        OCSVM(nu=0.9, kernel="sigmoid", contamination=0.01),
        
        OCSVM(nu=0.1, kernel="sigmoid", contamination=0.05),
        OCSVM(nu=0.3, kernel="sigmoid", contamination=0.05),
        OCSVM(nu=0.5, kernel="sigmoid", contamination=0.05),
        OCSVM(nu=0.7, kernel="sigmoid", contamination=0.05),
        OCSVM(nu=0.9, kernel="sigmoid", contamination=0.05),

        
        LOF(n_neighbors=1, metric='manhattan', contamination=0.005, n_jobs = n_jobs),
        LOF(n_neighbors=5, metric='manhattan', contamination=0.005, n_jobs = n_jobs),
        LOF(n_neighbors=15, metric='manhattan', contamination=0.005, n_jobs = n_jobs),
        LOF(n_neighbors=25, metric='manhattan', contamination=0.005, n_jobs = n_jobs),
        LOF(n_neighbors=50, metric='manhattan', contamination=0.005, n_jobs = n_jobs),
        LOF(n_neighbors=70, metric='manhattan', contamination=0.005, n_jobs = n_jobs),
        LOF(n_neighbors=100, metric='manhattan', contamination=0.005, n_jobs = n_jobs),
        
        LOF(n_neighbors=1, metric='manhattan', contamination=0.01, n_jobs = n_jobs),
        LOF(n_neighbors=5, metric='manhattan', contamination=0.01, n_jobs = n_jobs),
        LOF(n_neighbors=15, metric='manhattan', contamination=0.01, n_jobs = n_jobs),
        LOF(n_neighbors=25, metric='manhattan', contamination=0.01, n_jobs = n_jobs),
        LOF(n_neighbors=50, metric='manhattan', contamination=0.01, n_jobs = n_jobs),
        LOF(n_neighbors=70, metric='manhattan', contamination=0.01, n_jobs = n_jobs),
        LOF(n_neighbors=100, metric='manhattan', contamination=0.01, n_jobs = n_jobs),
        
        LOF(n_neighbors=1, metric='manhattan', contamination=0.05, n_jobs = n_jobs),
        LOF(n_neighbors=5, metric='manhattan', contamination=0.05, n_jobs = n_jobs),
        LOF(n_neighbors=15, metric='manhattan', contamination=0.05, n_jobs = n_jobs),
        LOF(n_neighbors=25, metric='manhattan', contamination=0.05, n_jobs = n_jobs),
        LOF(n_neighbors=50, metric='manhattan', contamination=0.05, n_jobs = n_jobs),
        LOF(n_neighbors=70, metric='manhattan', contamination=0.05, n_jobs = n_jobs),
        LOF(n_neighbors=100, metric='manhattan', contamination=0.05, n_jobs = n_jobs),
        
        LOF(n_neighbors=1, metric='euclidean', contamination=0.005, n_jobs = n_jobs),
        LOF(n_neighbors=5, metric='euclidean', contamination=0.005, n_jobs = n_jobs),
        LOF(n_neighbors=15, metric='euclidean', contamination=0.005, n_jobs = n_jobs),
        LOF(n_neighbors=25, metric='euclidean', contamination=0.005, n_jobs = n_jobs),
        LOF(n_neighbors=50, metric='euclidean', contamination=0.005, n_jobs = n_jobs),
        LOF(n_neighbors=70, metric='euclidean', contamination=0.005, n_jobs = n_jobs),
        LOF(n_neighbors=100, metric='euclidean', contamination=0.005, n_jobs = n_jobs),
        
        LOF(n_neighbors=1, metric='euclidean', contamination=0.01, n_jobs = n_jobs),
        LOF(n_neighbors=5, metric='euclidean', contamination=0.01, n_jobs = n_jobs),
        LOF(n_neighbors=15, metric='euclidean', contamination=0.01, n_jobs = n_jobs),
        LOF(n_neighbors=25, metric='euclidean', contamination=0.01, n_jobs = n_jobs),
        LOF(n_neighbors=50, metric='euclidean', contamination=0.01, n_jobs = n_jobs),
        LOF(n_neighbors=70, metric='euclidean', contamination=0.01, n_jobs = n_jobs),
        LOF(n_neighbors=100, metric='euclidean', contamination=0.01, n_jobs = n_jobs),
    
        LOF(n_neighbors=1, metric='euclidean', contamination=0.05, n_jobs = n_jobs),
        LOF(n_neighbors=5, metric='euclidean', contamination=0.05, n_jobs = n_jobs),
        LOF(n_neighbors=15, metric='euclidean', contamination=0.05, n_jobs = n_jobs),
        LOF(n_neighbors=25, metric='euclidean', contamination=0.05, n_jobs = n_jobs),
        LOF(n_neighbors=50, metric='euclidean', contamination=0.05, n_jobs = n_jobs),
        LOF(n_neighbors=70, metric='euclidean', contamination=0.05, n_jobs = n_jobs),
        LOF(n_neighbors=100, metric='euclidean', contamination=0.05, n_jobs = n_jobs),
    
        LOF(n_neighbors=1, metric='minkowski', contamination=0.005, n_jobs = n_jobs),
        LOF(n_neighbors=5, metric='minkowski', contamination=0.005, n_jobs = n_jobs),
        LOF(n_neighbors=15, metric='minkowski', contamination=0.005, n_jobs = n_jobs),
        LOF(n_neighbors=25, metric='minkowski', contamination=0.005, n_jobs = n_jobs),
        LOF(n_neighbors=50, metric='minkowski', contamination=0.005, n_jobs = n_jobs),
        LOF(n_neighbors=70, metric='minkowski', contamination=0.005, n_jobs = n_jobs),
        LOF(n_neighbors=100, metric='minkowski', contamination=0.005, n_jobs = n_jobs),
        
        LOF(n_neighbors=1, metric='minkowski', contamination=0.01, n_jobs = n_jobs),
        LOF(n_neighbors=5, metric='minkowski', contamination=0.01, n_jobs = n_jobs),
        LOF(n_neighbors=15, metric='minkowski', contamination=0.01, n_jobs = n_jobs),
        LOF(n_neighbors=25, metric='minkowski', contamination=0.01, n_jobs = n_jobs),
        LOF(n_neighbors=50, metric='minkowski', contamination=0.01, n_jobs = n_jobs),
        LOF(n_neighbors=70, metric='minkowski', contamination=0.01, n_jobs = n_jobs),
        LOF(n_neighbors=100, metric='minkowski', contamination=0.01, n_jobs = n_jobs),
        
        LOF(n_neighbors=1, metric='minkowski', contamination=0.05, n_jobs = n_jobs),
        LOF(n_neighbors=5, metric='minkowski', contamination=0.05, n_jobs = n_jobs),
        LOF(n_neighbors=15, metric='minkowski', contamination=0.05, n_jobs = n_jobs),
        LOF(n_neighbors=25, metric='minkowski', contamination=0.05, n_jobs = n_jobs),
        LOF(n_neighbors=50, metric='minkowski', contamination=0.05, n_jobs = n_jobs),
        LOF(n_neighbors=70, metric='minkowski', contamination=0.05, n_jobs = n_jobs),
        LOF(n_neighbors=100, metric='minkowski', contamination=0.05, n_jobs = n_jobs),
    ]
    
    randomness_flags.extend([False]*27)  # IForest
    randomness_flags.extend([False]*63)  # KNN
    randomness_flags.extend([False]*60)  # OCSVM
    randomness_flags.extend([False]*63)  # LOF
    return BASE_ESTIMATORS, randomness_flags