# Đại số tuyến tính cho Machine Learning

## 1.1. Sơ lược về ký hiệu

Các đại lượng vô hướng là các chữ thường in nghiêng vd: $x, y, N$. Các ma trận là chữ viết hoa in đậm vd: $\mathbf{X}, \mathbf{Y}$. Các vector được biểu diễn bởi chữ thường in đậm, mặc định là **vector cột** vd: $\mathbf{x}, \mathbf{y}$.

Lưu ý với vector: $\mathbf{x} = [x_1, x_2, x_3, ..., x_n]$ là vector hàng. Còn $\mathbf{x} = [x_1; x_2;...;x_n]$ là vector cột. Sự khác biệt ở đây là dấu (,) và dấu chấm phẩy.

Tương tự đối với ma trận. $\mathbf{X=[x_1, x_2, ... , x_n]}$ là ma trận mà các vector là cột xếp cạnh nhau từ trái sang phải. Trong khi $\mathbf{X = [x_1;x_2;...;x_n]}$ là ma trận mà các vector là hàng xếp chồng nhau.

Cho một ma trận $\mathbf{W}$, nếu không nói gì thêm thì $\mathbf{w_i}$ là vector cột thứ i của ma trận $\mathbf{W}$.

## 1.2. Chuyển vị và Hermitian

Cho ma trận/vector $\mathbf{A} \in \mathbb{R^{m \times n}}$, thì $\mathbf{B} \in \mathbb{R^{n \times m}}$ là chuyển vị của $\mathbf{A}$ nếu $b_{ij} = a_{ji}, \forall 1 \leq i \leq n, \forall 1 \leq j \leq m$.

$$\begin{gathered}\mathbf{x}=\begin{bmatrix}x_1\\x_2\\\vdots\\x_m\end{bmatrix}\Rightarrow\mathbf{x}^T=\begin{bmatrix}x_1 x_2 \ldots x_m\end{bmatrix};\\\mathbf{A}=\begin{bmatrix}a_{11}&a_{12}&\ldots&a_{1n}\\a_{21}&a_{22}&\ldots&a_{2n}\\\ldots&\ldots&\ddots&\ldots\\a_{m1}&a_{m2}&\ldots&a_{mn}\end{bmatrix}\Rightarrow\mathbf{A}^T=\begin{bmatrix}a_{11}&a_{21}&\ldots&a_{m1}\\a_{12}&a_{22}&\ldots&a_{m2}\\\ldots&\ldots&\ddots&\ldots\\a_{1n}&a_{2n}&\ldots&a_{mn}\end{bmatrix}\end{gathered}$$

Nếu $\mathbf{A}^T = \mathbf{A}$ ta nói $\mathbf{A}$ là ma trận đối xứng.

Trong trường hợp vector hoặc ma trận có số phức, việc lấy chuyển vị thường đi kèm với việc lấy liên hợp phức. Nghĩa là ngoài việc đổi vị trí của các phần tử ta còn lấy liên hợp của các phần tử đó. Tên gọi của phép chuyển vị này còn được gọi là *phép chuyển vị liên hợp* (conjugate transpose), thường được ký hiệu bằng $H$ thay cho $T$. Chuyển vị liên hợp của $\mathbf{A}$ là $\mathbf{A}^H$.

Cho $\mathcal{A}\in\mathbb{C}^{m\times n}$, ta nói $\mathcal{B}\in\mathbb{C}^{n\times m}$ là chuyển vị liên hợp cűa $\mathbf{A}$ nếu $b_{ij}=\overline{a_{ji}},\quad\forall1\leq$ $i\leq n,1\leq j\leq m$, trong đó $\bar{a}$ là liên hiêp phúc cűa $a.$

Ví dụ:

$$\mathbf{A}=\begin{bmatrix}1+2i&3-4i\\i&2\end{bmatrix}\Rightarrow\mathbf{A}^H=\begin{bmatrix}1-2i&-i\\3+4i&2\end{bmatrix}$$

Nếu chuyển vị liên hợp của một ma trận vuông phức bằng với chính nó, $\mathbf{A}^H = \mathbf{A}$ thì ma trận $\mathbf{A}$ gọi là *Hermitian*.

## 1.3. Phép nhân hai ma trận

Cho $\mathbf{A} \in \mathbb{R}^{m \times n}, \mathbf{B} \in \mathbb{R}^{n \times p}$, tích của hai ma trận là $\mathbf{C = AB} \in \mathbb{R}^{m \times p}$. Trong đó mỗi phần tử của $\mathbf{C}$ được tính bởi:

$$c_{ij} = \sum_{k=1}^{n},\ \forall 1 \leq i \leq m, 1 \leq j \leq p$$

Để nhân ma trận thì số cột của ma trận 1 phải bằng số hàng ma trận 2.

Có các tính chất sau:

- Phép nhân ma trận không có tính chất giao hoán.
- Phép nhân ma trận có tính chất kết hợp: $\mathbf{(AB)C=A(BC)=ABC}$
- Phép nhân ma trận có tính phân phối với phép cộng: $\mathbf{A(B+C)=AB + AC}$
- Chuyển vị của tích bằng tích chuyển vị ngược lại: $\mathbf{(AB)^T = B^T A^T}$. Điều này cũng xảy ra với chuyển vị liên hợp.

Tích trong, hay tích vô hướng (inner product) của hai vector $\mathbf{x, y} \in \mathbb{R}^n$ là:

$$\mathbf{x^Ty = y^Tx} = \sum_{i=1}^n x_iy_i$$

Nếu tích vô hướng của hai vector khác không bằng không thì hai vector này *trực giao* (orthogonal).

Phép nhân của một ma trận $\mathbf{A} \in \mathbb{R}^{m \times n}$ và một vector $\mathbf{x} \in \mathbb{R}^n$ là một vector $\mathbf{b} \in \mathbb{R}^m$\

$$\mathbf{Ax=b},\ \text{với } \mathbf{A_{i:}x}$$

Ví dụ:

$$\left[\begin{matrix}
    1\   2 \\
    2\   3
\end{matrix}\right] \times \left[ \begin{matrix}
    2 \\
    2
\end{matrix}\right] = \left[ \begin{matrix}
    6 \\
    10
\end{matrix} \right]$$

Phép nhân từng thành phần gọi là *tích Hadamard*. Là lấy từng phần tử của ma trận 1 nhân với từng phần tử ma trận 2 ký hiệu là $\mathbf{C}=\mathbf{A}\odot\mathbf{B}$.

## 1.4. Ma trận đơn vị và ma trận nghịch đảo

### 1.4.1 Ma trận đơn vị

Đường chéo chính của ma trận là tập hợp các điểm có chỉ số hàng và cột bằng nhau. Nếu $\mathbf{A} \in \mathbb{R}^{m \times n}$ thì đường chéo chính của $\mathbf{A}$ là $\set{a_{11}, a_{22}, ..., a_{pp}}$ với $p = min\{m,n\}$ 

Vậy ma trận đơn vị là ma trận không nhưng có đường chéo chính bằng 1. Ký hiệu $I_n \in \mathbb{R^n}$. Ví dụ: 

$$\mathbf{I}_3=\begin{bmatrix}1 0 0\\0 1 0\\0 0 1\end{bmatrix},\quad\mathbf{I}_4=\begin{bmatrix}1 0 0 0\\0 1 0 0\\0 0 1 0\\0 0 0 1\end{bmatrix}$$

Tính chất đặc biệt $\mathbf{AI} = \mathbf{IA} = \mathbf{A}$

### 1.4.2. Ma trận nghịch đảo

Cho ma trận vuông $\mathbf{A} \in \mathbb{R}^{n \times n}$, ma trận $\mathbf{B} \in \mathbb{R}^{n \times n}$ là ma trận nghịch đảo của $\mathbf{A}$ khi $\mathbf{AB = I_n}$. Lúc đó $\mathbf{A}$ khả nghịch và $\mathbf{B}$ là ma trạn nghịch đảo của $\mathbf{A}$. Nếu không tồn tại ma trận $\mathbf{B}$ thỏa mãn điều trên thì $\mathbf{A}$ không khả nghịch. Ký hiệu $\mathbf{B = A^{-1}}$

Ma trận nghịch đảo dùng để giải hệ phương trình tuyến tính $\mathbf{Ax = b}$. Nếu $\mathbf{A}$ khả nghịch thì $\mathbf{x}$ có nghiệm duy nhất là $\mathbf{A^{-1}b}$.

Nếu $\mathbf{A}$ không khả nghịch, thậm chí không vuông, phương trình tuyến tính trên có thể vô nghiệm hoặc có vô số nghiệm.

Nếu $\mathbf{A,B}$ khả nghịch thì $\mathbf{(AB)^{-1}}=\mathbf{B^{-1}A^{-1}}$.

## 1.5. Một vài ma trận đặc biệt

### 1.5.1. Ma trận đường chéo

Là ma trận mà các thành phần khác không chỉ nằm trên đường chéo chính. Có thể áp dụng lên ma trận không vuông. Ma trận không, ma trận đơn vị là các ma trận đường chéo.

Ký hiệu $diag(a_{11}, a_{22}, ... a_{mm})$ là cho ma trận đường chéo vuông. Tích, tổng của hai ma trận đường chéo vuông cùng bậc là một ma trận đường chéo. Một ma trận đường chéo vuông là khả nghịch khi và chỉ khi mọi phần tử trên đường chéo chính của nó khác không. Nghịch đảo của một ma trận đường chéo khả nghịch cũng là một ma trận đường chéo. Cụ thể $(diag(a_{11}, a_{22}, ... a_{mm}))^{-1} = diag(a_{11}^{-1}, a_{22}^{-1}, ... a_{mm}^{-1})$.

### 1.5.2. Ma trận tam giác

Một ma trận vuông được gọi là ma trận tam giác trên nếu tất cả các thành phần nằm phía dưới đường chéo chính bằng 0. Tương tự, một ma trận vuông được gọi là ma trận tam giác dưới nếu tất cả các thành phần nằm phía trên đường chéo chính bằng 0.

## 1.6. Định thức

### 1.6.1 Định nghĩa

Định thức của một ma trận vuông bậc $n$ $\mathbf{A}$ ký hiệu là $\det(\mathbf{A})$.

Với $n=1$ định thức của ma trận = phần tử duy nhất cua ma trận đó.

Với $n > 1$:

$$\mathbf{A}=\begin{bmatrix}a_{11}&a_{12}&\ldots&a_{1n}\\a_{21}&a_{22}&\ldots&a_{2n}\\\ldots&\ldots&\ddots&\ldots\\a_{n1}&a_{n2}&\ldots&a_{nn}\end{bmatrix}\Rightarrow\det(\mathbf{A})=\sum_{j=1}^n(-1)^{i+j}a_{ij}\det(\mathbf{A}_{ij})$$

Trong đó $\mathbf{A_{ij}}$ là phần bù đại số của $\mathbf{A}$ ứng với phần tử ở hàng $i$, cột $j$. Phần bù đại số này là ma trận con của $\mathbf{A}$ bằng cách bỏ đi hàng $i$, cột $j$ của ma trận $\mathbf{A}$.

### 1.6.2. Tính chất

- det($\mathbf{A}$) = det($\mathbf{A^T}$)
- Định thức của ma trận đường chéo bằng tích các phần tử trên đường chéo chính.
- Định thức của ma trận đơn vị bằng 1.
- Định thức của ma trận tích bằng tích định thức: $det(\mathbf{AB}) = det(\mathbf{A})det(\mathbf{B})$ với $\mathbf{A, B}$ cùng chiều.
- Nếu ma trận có một hàng hoặc một cột là vector không thì định thức bằng không.
- Ma trận khả nghịch khi và chỉ khi định thức khác 0.
- Nếu một ma trận khả nghịch, định thức của ma trận nghịch đảo bẳng nghịch đảo định thức của nó

$$det(\mathbf{A^{-1}}) = \frac{1}{det(\mathbf{A})} \text{ vì } \det(\mathbf{A})\det(\mathbf{A}^{-1})=\det(\mathbf{A}\mathbf{A}^{-1})=\det(\mathbf{I})=1$$ 

## 1.7 Tổ hợp tuyến tính, không gian sinh

### 1.7.1. Tổ hợp tuyến tính

Cho các vector khác không $\mathbf{a_1, ..., a_n} \in \mathbb{R}^{m}$ và các số thực $x_1, \dots, x_n \in \mathbb{R}$, vector:

$$\mathbf{b} = x_1a_1 + x_2a_2 + ... + x_na_n$$

được gọi là một *tổ hợp tuyến tính* của $\mathbf{a_1, ..., a_n}$. Xét ma trận $\mathbf{A} = [\mathbf{a_1, a_2, ..., a_n}] \in \mathbb{R^{m \times n}}$. Biểu thức trên sẽ được viết thành $\mathbf{b=Ax}$. Ta nói $\mathbf{b}$ là một tổ hợp tuyến tính các cột của $\mathbf{A}$.

Tập hợp các vector có thể biểu diễn được dưới dạng tổ hợp tuyến tính của một hệ vector được gọi là *không gian sinh* của hệ vector đó. Không gian sinh của một hệ vector được ký hiệu là $span(\mathbf{a_1, ..., a_n})$. Nếu phương trình:

$$0 = x_1\mathbf{a_1} + ... + x_n\mathbf{a_n}$$

có nghiệm duy nhất $x_1 = x_2 = ... = x_n = 0$ thì hệ $\set{\mathbf{a_1, a_2, ..., a_n}}$ *độc lập tuyến tính*. Ngược lại nếu tồn tại $x_1 \neq 0$ sao cho phương trình trên thỏa mãn thì hệ đó *phụ thuộc tuyến tính*.

### 1.7.2. Tính chất

- Một hệ *phụ thuộc tuyến tính* khi và chỉ khi tồn tại một vector trong hệ đó là tổ hợp tuyến tính của các vector còn lại.
- Tập con khác rỗng của một hệ *độc lập tuyến tính* là một hệ *độc lập tuyến tính*.
- Các cột của ma trận khả nghịch tạo thành một hệ độc lập tuyến tính.

Giả sử ma trận $\mathbf{A}$ khả nghịch, phương trình $\mathbf{Ax = 0}$ có nghiệm duy nhất $\mathbf{x = A^{-1}0 = 0}$. Vậy $\mathbf{A}$ là hệ độc lập tuyến tính.

- Nếu $\mathbf{A}$ là ma trận cao, nghĩa là số hàng nhiều hơn số cột, $m > n$, tồn tại vector $\mathbf{b}$ sao cho phương trình $\mathbf{Ax=b}$ vô nghiệm.
- Nếu $n > m$, $n$ vector bất kỳ trong không gian $m$ chiều tạo thành một hệ phụ thuộc tuyến tính.

### 1.7.3. Cơ sở của một không gian

Một hệ các vector $\set{\mathbf{a_1, ..., a_n}}$ trong không gian vector $m$ chiều $V = \mathbb{R}^m$ được gọi là một cơ sở nếu hai điều kiện sau thỏa mãn:

- $V = span(\mathbf{a_1, ..., a_n})$
- $\set{\mathbf{a_1, a_2, ..., a_n}}$

Khi đó, mọi vector $\mathbf{b} \in V$ đều có thể biểu diễn duy nhất dưới dạng tổ hợp tuyến tính của các $\mathbf{a_i}$. Từ hai tính chất cuối ở mục 1.7.2, ta có thể suy ra rằng $m = n$.

### 1.7.4. Range và Null space

Với mỗi $\mathbf{A} \in \mathbb{R}^{m \times n}$, có hai không gian con quan trọng ứng với ma trận này.

*Range* của $\mathbf{A}$, ký hiệu $\mathcal{R}(\mathbf{A})$, được định nghĩa bởi

$$\mathcal{R}(\mathbf{A}) = \set{\mathbf{y} \in \mathbb{R}^m : \exists \mathbf{x} \in \mathbb{R}^n, \mathbf{Ax=y}}$$

$\mathcal{R}(\mathbf{A})$ là không gian sinh của các cột của $\mathbb{A}$. $\mathcal{R}(\mathbf{A})$ là một không gian con của $\mathbb{R}^n$ với số chiều bằng số lớn nhất của các cột độc lập tuyến tính của $\mathbf{A}$.

*Null* của $\mathbf{A}$, ký hiệu $\mathcal{N}(\mathbf{A})$, được định nghĩa bởi

$$\mathcal{N}(\mathbf{A}) = \set{\mathbf{x} \in \mathbb{R}^n : \mathbf{Ax=0}}$$

Mỗi vector trong $\mathcal{N}(\mathbf{A})$ tương ứng với bộ các hệ số làm cho tổ hợp tuyến tính các cột của $\mathbf{A}$ bằng vector 0. 

Có 

$$dim(\mathcal{R}(\mathbf{A})) + dim(\mathcal{N}(\mathbf{A})) = n$$

## 1.8. Hạng của ma trận

Hạng của ma trận $\mathbf{A} \in \mathbb{R}^{m \times n}$, ký hiệu là $rank(\mathbf{A})$, được định nghĩa là số lượng lớn nhất các cột của nó tạo thành một hệ độc lập tuyến tính. (số cột khác không)

Một số tính chất quan trọng của hạng

- Một ma trận có hạng bằng 0 khi và chỉ khi đó là ma trận không.
- Hạng của ma trận bằng hạng của ma trận chuyển vị. $rank(\mathbf{A}) = rank(\mathbf{A^T})$.
- Hạng của ma trận không lớn hơn số hàng hoặc số cột. Nếu $\mathbf{A} \in \mathbb{R}^{n \times m}$ thì $rank(\mathbf{A}) \leq min(m,n)$.
- Hạng của một tích không vượt quá hạng của một nhân tử. $rank(\mathbf{AB}) \leq min(rank(\mathbf{A}), rank(\mathbf{B}))$.
- Hạng của tổng không vượt quá tổng các hạng. $rank(\mathbf{A + B}) \leq rank(\mathbf{A}) + rank(\mathbf{B})$.

## 1.9. Hệ trực chuẩn, ma trận trực giao