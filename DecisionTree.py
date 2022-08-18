import numpy as np
import math
from Criterion import *

class DecisionNode():
    def __init__(self, feature_i=None, threshold=None, value=None, left_subtree=None, right_subtree=None):
        self.feature_i = feature_i                # Thuộc tính (thứ i) dùng để phân chia nút thành các nhánh
        self.threshold = threshold                # Ngưỡng để phân chia thuộc tính
        self.value = value                        # Kết quả dự đoán nếu nút đang xét là nút lá
        self.left_subtree = left_subtree          # cây con trái của nút đang xét
        self.right_subtree = right_subtree        # cây con phải của nút đang xét

# Class cha của RegressionTree và ClassificationTree
class DecisionTree():
    def __init__(self, min_sample_split=2, min_impurity=1e-7, max_depth=float("inf"), criterion='default'):
        self.root = None                          # nút gốc của cây
        self.min_sample_split = min_sample_split  # (Pre-pruning) Số lượng mẫu dữ liệu tối thiểu trong 1 nút
                                                  # cần thiết để tiếp tục xây dựng cây con
        self.min_impurity = min_impurity          # (Pre-pruning) Độ tinh khiết tối thiểu mỗi cần đạt được ở nút đó để tiếp tục phân chia
        self.max_depth = max_depth                # (Pre-pruning) Độ sâu tối đa cho phép để xây dựng cây
        self.criterion = criterion                # Hàm dùng đánh giá hiểu quả của sự phân chia ở 1 nút 
                                                  # (Classification: 'Gini' hoặc 'Entropy', Regression: 'MSE' hoặc 'MSA') 
        self._impurity_calculation = None         # Hàm dùng tính toán hiệu quả phân chia
        self._leaf_value_calculation = None       # Hàm dùng xác định giá trị dự đoán (nếu là nút lá)

    def fit(self, X, y, loss=None):
        """ Sử dụng dữ liệu huấn luyện và nhãn để xây dựng cây
        X (training data) - numpy array, có chiều là (n_samples, n_features)
        y (label data) - numpy array, có chiều là (n_samples, 1)
        """
        
        self.root = self._build_tree(X, y)
    
    def _build_tree(self, X, y, current_depth=0):

        largest_impurity = 0
        best_criteria = None    
        best_sets = None        

        # Kiểm tra xem dữ liệu nhãn có chiều đúng chưa (n_feature, 1)
        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)

        Xy = np.concatenate((X, y), axis=1)

        n_samples, n_features = np.shape(X)
        if n_samples >= self.min_sample_split and current_depth <= self.max_depth:
            # Thực hiện tính độ tinh khiết cho mỗi thuộc tính và tìm ra thuộc tính tốt nhất để làm ngưỡng phân chia nút
            for feature_i in range(n_features):
                # lấy tất cả giá trị của các thuộc tính trong X và chọn ra các thuộc tính có giá trị không trùng lặp
                feature_values = np.expand_dims(X[:, feature_i], axis=1) 
                unique_values = np.unique(feature_values)

                # Duyệt qua tất cả thuộc tính có giá trị duy nhất và tính độ tinh khiết
                for threshold in unique_values:
                    # Phân chia dữ liệu thành 2 tập con dựa trên ngưỡng (threshold) 
                    Xy1, Xy2 = divide_on_feature(Xy, feature_i, threshold)

                    if len(Xy1) > 0 and len(Xy2) > 0:
                        # lấy ra y của 2 tập dữ liệu con
                        y1 = Xy1[:, n_features:]
                        y2 = Xy2[:, n_features:]

                        # Tính độ tinh khiết
                        impurity = self._impurity_calculation(y, y1, y2)

                        # Nếu ngưỡng đang xét có hiệu quả phân chia tốt hơn thì ta lưu trữ
                        # thuộc tính đang xét và các tập dữ liệu con
                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {'feature_i': feature_i, 'threshold': threshold}
                            best_sets = {
                                'leftX': Xy1[:, :n_features],     # X của cây con trái
                                'lefty': Xy1[:, n_features:],     # y của cây con trái
                                'rightX': Xy2[:, :n_features],    # X của cây con phải
                                'righty': Xy2[:, n_features:]     # y của cây con phải
                            }

        if largest_impurity > self.min_impurity:
            # Tiếp tục xây dựng các cây con trái và phải nếu các điều kiện vẫn thỏa mãn
            left_subtree = self._build_tree(best_sets['leftX'], best_sets['lefty'], current_depth+1)
            right_subtree = self._build_tree(best_sets['rightX'], best_sets['righty'], current_depth+1)
            return DecisionNode(feature_i=best_criteria['feature_i'], threshold=best_criteria['threshold'],
                                left_subtree=left_subtree, right_subtree=right_subtree)

        # Cây là cây đầy đủ hoặc các điều kiện không được thỏa mãn => ta đến nút lá
        leaf_value = self._leaf_value_calculation(y)

        return DecisionNode(value=leaf_value)

    def predict_value(self, x, tree=None):
        """ Thực hiện tìm kiếm đệ quy theo các ngưỡng phân chia của cây 
        và lấy giá trị dự đoán theo giá trị nút lá mà ta tìm đến được. """

        if tree is None:
            tree = self.root

        # Nếu nút đang xét có giá trị thì ta đang ở nút lá => trả về giá trị dự đoán
        if tree.value is not None:
            return tree.value

        # chọn thuộc tính mà ta sẽ xem xet
        feature_value = x[tree.feature_i]

        # Xác định xem ta sẽ tìm kiếm theo cây con trái hay cây con phải
        branch = tree.right_subtree
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.threshold:
                branch = tree.left_subtree
        elif feature_value == tree.threshold:
            branch = tree.left_subtree

        # Bước đệ quy tìm kiếm trên cây con
        return self.predict_value(x, branch)

    def predict(self, X):
        """ Lấy kết quả dự đoán của từng mẫu dữ liệu và trả về một dãy kết quả dự đoán 
        cho toàn bộ dữ liệu."""
        y_pred = [self.predict_value(sample) for sample in X]
        return y_pred

    
class RegressionTree(DecisionTree):
    def _calculate_MSE(self, y, y1, y2):
        MSE_total = Mean_Square_Error(y)
        MSE_left = Mean_Square_Error(y1)
        MSE_right = Mean_Square_Error(y2)
        frac_1 = len(y1) / len(y)
        frac_2 = len(y2) / len(y)

        impurity_improvement = MSE_total - (frac_1 * MSE_left + frac_2 * MSE_right)

        return impurity_improvement
    
    def _calculate_MSA(self, y, y1, y2):

        MSA_total = Mean_Square_Error(y)
        MSA_left = Mean_Square_Error(y1)
        MSA_right = Mean_Square_Error(y2)
        frac_1 = len(y1) / len(y)
        frac_2 = len(y2) / len(y)

        impurity_improvement = MSA_total - (frac_1 * MSA_left + frac_2 * MSA_right)

        return impurity_improvement

    def _mean_of_y(self, y):
        value = np.mean(y, axis=0)
        return value if len(value) > 1 else value[0]

    def fit(self, X, y):
        if self.criterion == 'default':
            self._impurity_calculation = self._calculate_MSE
        elif self.criterion == 'MSA':
            self._impurity_calculation = self._calculate_MSA

        self._leaf_value_calculation = self._mean_of_y
        super(RegressionTree, self).fit(X, y)

class ClassificationTree(DecisionTree):
    def _calculate_entropy(self, y, y1, y2):
        
        p = len(y1) / len(y)
        entropy = Entropy(y)
        impurity_improvement = entropy - p * Entropy(y1) - (1-p) * Entropy(y2)
        return impurity_improvement
    
    def _calculate_gini_index(self, y, y1, y2):

        p = len(y1) / len(y)
        gini = Gini_index(y)
        impurity_improvement = gini - p * Gini_index(y1) - (1-p) * Gini_index(y2)
        return impurity_improvement  

    def _majority_vote(self, y):
        most_common = None
        max_count = 0
        for label in np.unique(y):
            # đếm số dự đoán với từng nhãn một.
            count = len(y[y==label])
            if count > max_count:
                most_common = label
                max_count = count
        return most_common
    
    def fit(self, X, y):
        if self.criterion == 'default':
            self._impurity_calculation = self._calculate_gini_index
        elif self.criterion == 'Entropy':
            self._impurity_calculation = self._calculate_entropy
            
        self._leaf_value_calculation = self._majority_vote
        super(ClassificationTree, self).fit(X, y)