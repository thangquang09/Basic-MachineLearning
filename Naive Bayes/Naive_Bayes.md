# Bộ phân loại Naive Bayes - Naive Bayes Classifier

## 1. Sơ lược

Xét một bài toán phân loại với $C$ nhãn khác nhau. Thay vì tìm chính xác nhãn của mỗi điểm dữ liệu $\mathbf{x} \in \mathbb{R}^d$, ta có thể tìm xác suất $\mathbb{P}(y=c|\mathbf{x})$, hoặc $\mathbb{P}(c|\mathbf{x})$. Gọi là xác suất để đầu ra là nhãn $c$ biết đầu vào là $\mathbf{x}$. Ta xác định nhãn của mỗi điểm dữ liệu bằng cách chọn ra nhãn có xác xuất rơi vảo cao nhất:
$$c = \arg\max_{c \in \{1, \dots C\}} \mathbb{P}(c|\mathbf{x}) \tag{1}$$
Ta sử dụng công thức Bayes:
>[!Note]
>Công thức Bayes:
>Cho họ biến cố $B_{1},B_{2},\dots,B_{n}$ sao cho hội của chúng là $\Omega$ và giao của đôi một biến cố là rỗng. Khi đó với mọi biến cố $A$
>$$\mathbb{P}(B_{k}|A) = \frac{\mathbb{P}(B_{k} \cap A)}{\mathbb{P}(A)} = \frac{\mathbb{P}(A|B_{k})\times P(B_{k})}{P(A) = \sum_{j = 1}^n \mathbb{P}(B_{j})\times \mathbb{P}(A|B_{j})}$$

Ta viết $(1)$ thành:
$$c = \arg\max_{c} \mathbb{P}(c|\mathbf{x})=\arg\max_{c} \frac{\mathbb{P}(\mathbf{x}|c)\times \mathbb{P}(c)}{\mathbb{P}(\mathbf{x})}=\arg\max_{c}\mathbb{P}(\mathbf{x}|c)\times \mathbb{P}(c)\tag{2}$$
Dấu bằng thứ hai xảy ra theo công thức Bayes, dấu bằng thứ ba xảy ra vì $\mathbb{P}(\mathbf{x})$ ở mẫu số không phụ thuộc vào $c$. $\mathbb{P}(c)$ được hiểu là ==xác suất== để một điểm *bất kỳ* rơi vào nhãn $c$.

Nếu tập huấn luyện (training set) lớn, $\mathbb{P}(c)$ có thể được xác định bằng **MLE** (ước lượng hợp lý cực đại) - tỉ lệ giữa số điểm thuộc nhãn $c$ và số điểm trong tập huấn luyện. Nếu tập huấn luyện nhỏ, giá trị này xác định bằng **MAP** (ước lượng hậu nghiệm cực đại).

$\mathbb{P}(\mathbf{x}|c)$ là phân phối của các điểm dữ liệu trong nhãn $c$. Thành phần này khó tính toán vì $\mathbf{x}$ là biến ngẫu nhiên nhiều chiều. Nhằm đơn giản hóa việc tính toán, người ta giả sử các thành phần của biến ngẫu nhiên $\mathbf{x}$ độc lập với nhau khi biết $c$:
$$\mathbb{P}(\mathbf{x}|c)=\mathbb{P}(x_{1}, x_{2}, x_{3}, \dots, x_{d}|c) = \prod_{i=1}^d \mathbb{P}(x_{i}|c) \tag{3}$$
Giả thiết độc lập này quá chặt và trên thực tế ít có trường hợp này xảy ra. Tuy nhiên, giả thiết *ngây thơ (naive)* này đôi khi mang lại kết quả bất ngờ. Giả thiết về sự độc lập giữa các chiều dữ liệu này gọi là *naive Bayes*. Một phương pháp xác định nhãn của dữ liệu trên giả thiết này gọi là *naive Bayes Classifier (NBC)*.

Nhờ giả thiết độc lập, NBC huấn luyện và kiểm tra rất nhanh. Quan trọng với bài toán ==có dữ liệu lớn==.

Trong huấn luyện, $\mathbb{P}(c)$ và $\mathbb{P}(x_{i}|c), i = 1,\dots,d$ được tính toán dựa trên tập huấn luyện. Bằng **MLE hoặc MAP**

Trong kiểm tra, nhãn của điểm dữ liệu $\mathbf{x}$ được xác định bởi:
$$c = \arg\max_{c} \mathbb{P}(c) \times \prod_{i=1}^d\mathbb{P}(x_{i}|c) \tag{4}$$
>[!note]
>Khi $d$ lớn và xác xuất nhỏ thì vế trái sẽ là số cực kì nhỏ, khi tính toán có thể có sai số. Để giải quyết ta viết $(4)$ bằng phép tương đương lấy log 2 vế
>$$\log(c) = \arg\max_{c} \left( \log(\mathbb{P}(c)) +\sum_{i=1}^d\log(\mathbb{P(x_{i|c})}) \right) \tag{5}$$
>Việc này không ảnh hưởng tới kết quả vì hàm log là hàm đồng biến
trên tập số dương

*NBC* mang lại hiệu quả trong bài toán phân loại văn bản, ví dụ lọc tin nhắn hoặc email rác.

Việc tính toán $\mathbb{P}(x_{i}|c)$ phụ thuộc vào loại dữ liệu. Có 3 loại phân bố xác suất phổ biến là:

- *Gaussian naive Bayes*
- *multinomial naive Bayes*
- *Bernoulli Naive*

## 2. Các phân phối thường dùng

### **2.1 Gaussian naive Bayes**

Sử dụng chủ yếu trong loại dữ liệu mà các thành phần là ==biến liên tục==. Với mỗi chiều dữ liệu $i$, một nhãn $c$ thì $x_{i} \sim N(\mu_{ci}, \sigma^2_{ci})$:
$$\mathbb{P}(x_{i}|c) = \mathbb{P}(x_{i}|\theta = \mu_{ci}, \sigma^2_{ci}) = \frac{1}{\sqrt{ 2\pi \sigma^2_{ci} }}\exp\left( -\frac{(x_{i}-\mu_{ci})^2}{2\sigma_{ci}^2} \right) \tag{6}$$
Trong đó, $\theta =\{ \mu_{ci}, \sigma^2_{ci}\}$ được xác định bằng **MLE** dựa trên các điểm trong tập huấn luyện mang nhãn $c$.

### **2.2 Multinomial naive Bayes**

Sử dụng chủ yếu trong bài toán phân loại văn bản mà vector đặc trưng được xây dựng dựa trên [bag of Words](bag of Words). Lúc này mỗi văn bản được biểu diễn bởi một vector độ dài $d$ là số từ trong từ điển. Giá trị của thành phần thứ $i$ trong mỗi vector là số lần từ thứ $i$ xuất hiện trong văn bản đó. Vậy $\mathbb{P}(x_{i}|c)$ tỉ lệ với tần suất của từ thứ $i$ xuất hiện trong văn bản có nhãn $c$. Giá trị này được tính:

$$\lambda_{ci} = \frac{N_{ci}}{N_{c}} \tag{7}$$
Trong đó:

- $N_{ci}$ là tổng số lần từ thứ $i$ xuất hiện trong các văn bản của nhãn $c$. Là tổng tất cả thành phần thứ $i$ trong vector đặc trưng ứng với nhãn $c$.
- $N_{c}$ là tổng số từ, kể cả lặp, xuất hiện trong nhãn $c$. Hay $N_{c}$ là tổng độ dài của tất cả văn bản thuộc nhãn $c$. $\implies N_{c} = \sum_{i=1}^dN_{ci} \implies \sum_{i=1}^d \lambda_{c_{i}}=1$

>[!Warning]
>Nếu có một từ mới chưa bao giờ xuất hiện trong nhãn $c$ thì $(7)$ sẽ bằng không. Khiến cho vế phải của $(4)$ bằng không bất kể $\mathbb{P}(c)$ lớn cở nào.
>Khắc phục: (Laplace Smoothing)
>$$\hat{\lambda_{ci}} = \frac{N_{ci} +{ \alpha}}{N_{c} + d\alpha} \tag{8}$$
>với $\alpha > 0$ thường chọn bằng $1$, để tránh tử số bằng không. Mẫu số + thêm $d\alpha$ để đảm bảo $\sum_{i=1}^{d} \hat{\lambda_{ci}} = 1$

### **2.3 Bernoulli Naive Bayes**

### **3.1 Bắc hay Nam**

Được áp dụng cho loại dữ liệu mà mỗi thành phần là một giá trị nhị phân. Ví dụ, thay vì đếm tổng số lần xuất hiện của từ nào đó, ta chỉ cần xác định từ đó có xuất hiện hay không.

Khi đó,

$$\mathbb{P}(x_{i}|c) = \mathbb{P}(i|c)^{x_{i}}\times(1-\mathbb{P}(i|c))^{1-x_{i}} \tag{9}$$

$\mathbb{P}(i|c)$ là xác suất từ thứ i xuất hiện trong văn bản của class $c$, $x_{i}$ mang giá trị 1 hoặc 0 tùy vào từ thứ $i$ có xuất hiện hay không.

## 3. Ví dụ

![[Pasted image 20240209172137.png]]

Trong tập huấn luyện d1, d2, d3, d4. Có nhãn là B hoặc N.
Ta dễ dàng đoán được d5 là B.

Có thể sử dụng NBC với phân phối *multinomial naive Bayes* hoặc *Bernoulli Naive*.
Có hai nhãn B và N. Ta cần tìm $\mathbb{P}(\text{B})$ và $\mathbb{P}(\text{N})$.
Dễ thấy:

$$\mathbb{P}(\text{B}) = \frac{3}{4},\ \ \ \ \ \ \ \mathbb{P}(\text{N}) = \frac{1}{4} \tag{10}$$
Tập hợp toàn bộ từ trong tập huấn luyện

$$V = \{\text{hanoi, pho, chaolong, buncha, omai, banhgio, saigon, hutiu, banhbo} \}$$
![[Pasted image 20240209172610.png]]
Minh họa pha huấn luyện và kiểm tra

Hình trên minh họa việc sử dụng phân phối *multinomial naive Bayes*. Trong đó Laplace Smoothing được sử dụng với $\alpha = 1$.

$$\mathbb{P}(B|d_{5}) = \frac{1.5 \times 10^{-4}}{1.5\times 10^{-4} + 1.75\times 10^{-5}} \approx 0.8955 \implies \mathbb{P}(N|d_{5}) \approx 1- 0.8955 = 0.1045$$

### **3.2 Thư viện scikit-learn**

Kiểm chứng lại dữ liệu trên với Python

file `naiveBayes_Bac_hay_Nam.ipynb`

### **3.3 Phân loại email rác với NBC**

 file `filter_spam_email.ipynb`

## 4. Thảo luận

- NBC (Bộ phân loại Naive Bayes) thường được sử dụng trong bài toán phân loại văn bản.
- NBC có thời gian huấn luyện rất nhanh và kiểm tra cũng rất nhanh. Do giả sử về tính độc lập giữa các thành phần.
- Nếu giả sử về tính độc lập được thỏa mãn. NBC thậm chí hoạt động tốt hơn SVM và Logistic Regresson khi có ít dữ liệu huấn luyện.
- NBC có thể hoạt động với các vector đặc trưng mà trong đó một phần là liên tục (sử dụng *Gaussian  naive Bayes*), một phần rời rạc (sử dụng *Multinomial* hoặc *Bernoulli*). Sự độc lập giữa các đặc trưng cho phép NBC có khả năng này.
- Laplace Smoothing được sử dụng để tránh trường hợp một từ trong tập kiểm tra chưa xuất hiện trong tập huấn luyện.
