# Task 2: Kernelize SVM
- Interface: cho phep use different kernel functions.
- Mercer's theorem: positive semi-definite matrix
- Radial Basis Function.

# TODO list:
- [ ] Implement svm kernel class
- [ ] Create testing program for svm kernel functions
- [ ] Create docs for svm kernel

Some useful note for my tasks, including the SVM algorithm, SVM kernels.
# SVM algorithm

Very Good explained: [MIT SVM algorithm](https://www.youtube.com/watch?v=_PwhiWxHK8o&ab_channel=MITOpenCourseWare), [MIT SVM examples](https://www.youtube.com/watch?v=6nDqY8MPLDM&ab_channel=MITOpenCourseWare)


## Defination
- Supervised learning methods used for [classification](https://scikit-learn.org/stable/modules/svm.html#svm-classification), [regression](https://scikit-learn.org/stable/modules/svm.html#svm-regression) and [outliers detection](https://scikit-learn.org/stable/modules/svm.html#svm-outlier-detection)
- Different [Kernel functions](https://scikit-learn.org/stable/modules/svm.html#svm-kernels) can be specified for the decision function
- Some type of SVM: SVC and NuSVC (one-versus-one), LinearSVC (sklearn)
- SVM còn được coi là maximum margin classifier.
- Sử dụng Karus Kuln Tucker để giải quyết bài toán tối ưu

## Some questions
- Why maximize margin?
Cần hiểu về khái niệm công bằng (các điểm được phân chia đồng đều - bằng nhau đến đường phân chia), và thịnh vượng (khoảng cách đến hyperplane cần max, khi đó cả 2 class đều vui)
- Các chứng minh dùng Largrage khá là khó hiểu, có cách nào intuative hơn không?
Phải hiểu, có link hay đây :v (below)
- Does Slack and C are hyperparameter?

## Support topics

### Linear Separator
- We have perceptron, why on earth we bother on this?
- In LS, can we pick 50 instead of 1 => its not a problem, choose any number (but equal in both side)

### Large Margin Classifier
- Basically, this is optimization problem
- Updating math function here .…
- Nếu gặp trường hợp point quá lỗi => sử dụng slack

### Largrange Mutiplier
- Constrain optimization problem [Explain Largrange](https://www.youtube.com/watch?time_continue=20&v=m-G3K2GPmEQ&feature=emb_logo&ab_channel=KhanAcademy)
- Gradients of curve is perpendicular
### Dual Problem
- w maybe in infinite dimensional space
- Vấn đề gì nếu các biết trong constraint là các biến cần tối ưu (nhiều) và ngược lại?
- Tại sao dùng Largrange functions cho dual problem?
Here is the answer [Duality Optimization](https://en.wikipedia.org/wiki/Duality_(optimization)#:~:text=The%20Lagrangian%20dual%20problem%20is,minimize%20the%20original%20objective%20function.)
- read goldstein? virtual force $\alpha$?  Primal ?
- Largrange functions (L) => Deravative at w, b need to be vanish => Plugging terms back to L
- Plugging back hiện tại đang chưa ra?

### Convex Programs
   - Primal optization problem
   - Lagrange function
   - First order optimality conditions in x
   - Solve for x and plug it back into L

# SVM kernel
- Tại sao cần kernel?
Các bài toán không phải lúc nào cũng linear separable => bài toán phân lớp Non-linear separable
- Phương pháp dùng SVM với dạng này:
   + Chuyển không gian dữ liệu về separable _input space_ => _feature space_
   + Áp dụng Linear SVM
- Tuy nhiên, không nhất thiết phải chuyển hẳn sang feature space để tính toán (bởi số chiều thường sẽ rất lớn)
=> Sử dụng các hàm nhân (kernel functions) `K`

`Một số kernel thường sử dụng: đa thức, Gaussian basis function, Sigmoidal`

- What is kernel trick?
Là việc tận dụng việc nhân tích vô hướng trong hàm optimization, vì vậy ta không cần phải tính giá trị trực tiếp của 1 điểm khi qua một không gian mới (việc này có thể rất tốn chi phí tính toán nếu số lượng chiều trong miền mới lớn!)

- How to choose the right kernel?
Updating ...

## Mercer Condition
- Hàm này được thiết kế sao cho hàm optimization là hàm lồi.

## Một số kernel thông dụng

### Linear

### Polinomial

###  RBF

### Sigmoid
