# Giải tích ma trận

<b style="font-size: 170%;">Table of Contents</b>
- [Giải tích ma trận](#giải-tích-ma-trận)
  - [2.1. Gradient của hàm trả về một số vô hướng](#21-gradient-của-hàm-trả-về-một-số-vô-hướng)
  - [2.2. Gradient của một hàm trả về vector](#22-gradient-của-một-hàm-trả-về-vector)
  - [2.3. Tính chất quan trọng](#23-tính-chất-quan-trọng)
    - [2.3.1. Quy tắc tích](#231-quy-tắc-tích)
    - [2.3.2. Quy tắc chuỗi](#232-quy-tắc-chuỗi)
  - [2.4. Gradient của các hàm số thường gặp](#24-gradient-của-các-hàm-số-thường-gặp)
    - [2.4.1. $f(\\mathbf{x}) = \\mathbf{a^Tx}$](#241-fmathbfx--mathbfatx)
    - [2.4.2. $f(\\mathbf{x}) = \\mathbf{Ax}$](#242-fmathbfx--mathbfax)
    - [2.4.3. $f(\\mathbf{x}) = \\mathbf{x^TAx}$](#243-fmathbfx--mathbfxtax)
    - [2.4.4. $f(\\mathbf{x}) = |\\mathbf{Ax - b}|\_2^2$](#244-fmathbfx--mathbfax---b_22)
    - [2.4.5. $f(\\mathbf{x}) = \\mathbf{a^Txx^Tb}$](#245-fmathbfx--mathbfatxxtb)
    - [2.4.6. $f(\\mathbf{X}) = trace(\\mathbf{AX})$](#246-fmathbfx--tracemathbfax)
    - [2.4.7. $f(\\mathbf{X}) = \\mathbf{a^TXb}$](#247-fmathbfx--mathbfatxb)
    - [2.4.8. $f(\\mathbf{X}) = |\\mathbf{X}|\_F^2$](#248-fmathbfx--mathbfx_f2)
    - [2.4.9. $f(\\mathbf{X}) = trace(\\mathbf{X^TAX})$](#249-fmathbfx--tracemathbfxtax)
    - [2.4.10. $f(\\mathbf{X}) = |\\mathbf{AX - B}|\_F^2$](#2410-fmathbfx--mathbfax---b_f2)
  - [2.5. Bảng các gradient thường gặp](#25-bảng-các-gradient-thường-gặp)
  - [2.6. Kiểm tra gradient](#26-kiểm-tra-gradient)
    - [2.6.1. Xấp xĩ đạo hàm của hàm một biến](#261-xấp-xĩ-đạo-hàm-của-hàm-một-biến)
    - [2.6.2. Xấp xĩ gradient của hàm nhiều biến](#262-xấp-xĩ-gradient-của-hàm-nhiều-biến)


Giả sử các gradient tồn tại trong chương!

## 2.1. Gradient của hàm trả về một số vô hướng

Gradient bậc nhất của một hàm số $f\mathbf{(x)}:\mathbb{R}^n \to \mathbb{R}$ theo $\mathbf{x}$, ký hiệu là $\nabla_\mathbf{x}f(\mathbf{x})$, được định nghĩa bởi:

$$\nabla_\mathbf{x}f(\mathbf{x})\triangleq\begin{bmatrix}\dfrac{\partial f(\mathbf{x})}{\partial x_1} \\ \dfrac{\partial f(\mathbf{x})}{\partial x_2} \\ \vdots \\ \dfrac{\partial f(\mathbf{x})}{\partial x_n}\end{bmatrix}\in\mathbb{R}^n$$

trong đó $\displaystyle\frac{\partial f(\mathbf{x})}{\partial x_i}$ là đạo hàm riêng của hàm số theo thành phần thứ $i$ của $\mathbf{x}$. Đạo hàm này được tính khi tất cả các biến ngoài $\mathbf{x}$ là hằng số. Nếu không có biến nào khác thì $\nabla_\mathbf{x} f(\mathbf{x})$ được viết gọn lại là $\nabla f(\mathbf{x})$. Gradient của hàm số là một vector cùng chiều với vector đang được lấy gradient. Nếu vector đó ở dạng cột thì gradient cũng ở dạng cột.

Gradient bậc hai của hàm số trên còn được gọi là *Hesse* (Hessian) và được định nghĩa như sau:

$$\begin{aligned}\nabla^2f(\mathbf{x})&\triangleq\begin{bmatrix}\frac{\partial^2f(\mathbf{x})}{\partial x_1^2}&\frac{\partial^2f(\mathbf{x})}{\partial x_1\partial x_2}\cdots\frac{\partial^2f(\mathbf{x})}{\partial x_1\partial x_n} \\ \frac{\partial^2f(\mathbf{x})}{\partial x_2\partial x_1}&\frac{\partial^2f(\mathbf{x})}{\partial x_2^2}\cdots\frac{\partial^2f(\mathbf{x})}{\partial x_2\partial x_n} \\ \vdots&\vdots&\ddots&\vdots \\ \frac{\partial^2f(\mathbf{x})}{\partial x_n\partial x_1}&\frac{\partial^2f(\mathbf{x})}{\partial x_n\partial x_2}\cdots\frac{\partial^2f(\mathbf{x})}{\partial x_n^2}\end{bmatrix}\in\mathbb{S}^n.\end{aligned}$$

Gradient của một hàm số $f(\mathbf{X}): \mathbb{R}^{n \times\ m} \to \mathbb{R}$ theo ma trận $\mathbf{X}$ định nghĩa là:

$$\nabla f(\mathbf{X})=\begin{bmatrix}\dfrac{\partial f(\mathbf{X})}{\partial x_{11}}&\dfrac{\partial f(\mathbf{X})}{\partial x_{12}}&\cdots\dfrac{\partial f(\mathbf{X})}{\partial x_{1m}} \\ \dfrac{\partial f(\mathbf{X})}{\partial x_{21}}&\dfrac{\partial f(\mathbf{X})}{\partial x_{22}}&\cdots\dfrac{\partial f(\mathbf{X})}{\partial x_{2m}} \\ \vdots&\vdots&\ddots&\vdots \\ \dfrac{\partial f(\mathbf{X})}{\partial x_{n1}}&\dfrac{\partial f(\mathbf{X})}{\partial x_{n2}}&\cdots\dfrac{\partial f(\mathbf{X})}{\partial x_{nm}}\end{bmatrix}\in\mathbb{R}^{n\times m}.$$

>Vậy Gradient của hàm số $f: \mathbb{R}^{n \times m} \to \mathbb{R}$ là một ma trận trong $\mathbb{R}^{n \times m}$

Cụ thể để tính gradient của một hàm $f: \mathbb{R}^{n \times m} \to \mathbb{R}$, ta tính đạo hàm riêng của hàm số đó theo từng thành phần khác được giả sử là hằng số. Tiếp theo ta sắp xếp các đạo hàm riêng tính được theo đúng thứ tự trong ma trận.

*Ví dụ:* Xét hàm số $f: \mathbb{R}^2 \to \mathbb{R},\ f(\mathbf{x}) = x_1^2 + 2x_1x_2 + sin(x_1) + 2$. Gradient bậc 1 theo $\mathbf{x}$ của hàm số đó là:

$$\nabla f(\mathbf{x})=\begin{bmatrix}\dfrac{\partial f(\mathbf{x})}{\partial x_1}\\\dfrac{\partial f(\mathbf{x})}{\partial x_2}\end{bmatrix}=\begin{bmatrix}\\2x_1+2x_2+\cos(x_1) \\ \\ 2x_1\end{bmatrix}$$

Gradient bậc 2 theo $\mathbf{x}$, hay Hesse là:

$$\nabla^2f(\mathbf{x})=\begin{bmatrix}\dfrac{\partial^2f(\mathbf{x})}{\partial x_1^2}&\dfrac{\partial f^2(\mathbf{x})}{\partial x_1\partial x_2} \\ \dfrac{\partial^2f(\mathbf{x})}{\partial x_2\partial x_1}&\dfrac{\partial f^2(\mathbf{x})}{\partial x_2^2}\end{bmatrix}=\begin{bmatrix}\\2-\sin(x_1)&2\\ \\ 2&0\end{bmatrix}$$

>Chú ý: Hesse luôn là ma trận đối xứng.

## 2.2. Gradient của một hàm trả về vector

Xét một hàm trả về vector với đầu vào là một số thực $v(x) : \mathbb{R} \to \mathbb{R}^n$:

$$v(x)=\begin{bmatrix}v_1(x) \\ v_2(x) \\ \vdots\\v_n(x)\end{bmatrix}.$$

Gradient của hàm số này theo $x$ là một vector hàng như sau:

$$\nabla v(x)\triangleq\left[\frac{\partial v_1(x)}{\partial x}\frac{\partial v_2(x)}{\partial x}\ldots\frac{\partial v_n(x)}{\partial x}\right].$$

Gradient bậc hai của hàm số này có dạng:

$$\nabla^2v(x)\triangleq\left[\frac{\partial^2v_1(x)}{\partial x^2}\frac{\partial^2v_2(x)}{\partial x^2}\ldots\frac{\partial^2v_n(x)}{\partial x^2}\right].$$

*Ví dụ:* Cho một vector $\mathbf{a} \in \mathbb{R}^n$ và một hàm trả về vector $v(x) = x\mathbf{a}$, gradient và Hesse của nó lần lượt là:

$$\nabla v(x)=\mathbf{a}^T,\quad\nabla^2v(x)=\mathbf{0}\in\mathbb{R}^{1\times n}.$$

Xét một hàm trả về vector với đầu vào là một vector $h(\mathbf{x}):\mathbb{R}^k \to \mathbb{R}^n$, gradient của nó là:

$$\nabla h(\mathbf{x})\triangleq\begin{bmatrix}\dfrac{\partial h_1(\mathbf{x})}{\partial x_1}&\dfrac{\partial h_2(\mathbf{x})}{\partial x_1}&\cdots\dfrac{\partial h_n(\mathbf{x})}{\partial x_1}\\\dfrac{\partial h_1(\mathbf{x})}{\partial x_2}&\dfrac{\partial h_2(\mathbf{x})}{\partial x_2}&\cdots\dfrac{\partial h_n(\mathbf{x})}{\partial x_2}\\\vdots&\vdots&\ddots&\vdots\\\dfrac{\partial h_1(\mathbf{x})}{\partial x_k}&\dfrac{\partial h_2(\mathbf{x})}{\partial x_k}&\cdots\dfrac{\partial h_n(\mathbf{x})}{\partial x_k}\end{bmatrix}=\begin{bmatrix}\nabla h_1(\mathbf{x})\ldots\nabla h_n(\mathbf{x})\end{bmatrix}\in\mathbb{R}^{k\times n}$$

>Gradient của hàm số $g: \mathbb{R}^m \to \mathbb{R}^n$ là một ma trận thuộc $\mathbb{R}^{m \times n}$

Trước khi đến phần tính gradient của các hàm số thường gặp, chúng ta cần biết một số tính chất quan trọng khá giống với gradient hàm một biến.

## 2.3. Tính chất quan trọng

### 2.3.1. Quy tắc tích

Giả sử các biến đầu vào là một ma trận và các hàm số có chiều phù hợp để phép nhân ma trận thực hiện được. Ta có:

$$\nabla(f(\mathbf{X})^Tg(\mathbf{X})) = (\nabla f(\mathbf{X}))g(\mathbf{X}) + (\nabla g(\mathbf{X}))f(\mathbf{X})$$

Quy tắc này tương tự như quy tắc tính đạo hàm tích trong 1 chiều với $f,g : \mathbb{R} \to \mathbb{R}$

$$(f(x)g(x))' = f'(x)g(x) + g'(x)f(x)$$

Lưu ý rằng tính chất giao hoán không còn đúng với vector và ma trận, vì vậy:

$$\nabla(f(\mathbf{X})^Tg(\mathbf{X})) \neq g(\mathbf{X})(\nabla f(\mathbf{X})) + f(\mathbf{X})(\nabla g(\mathbf{X}))$$

Biểu thức bên phải có thể không xác định vì chiều của ma trận lệch nhau.

### 2.3.2. Quy tắc chuỗi

Quy tắc chuỗi được áp dụng khi tính gradient của hàm hợp

$$\nabla_\mathbf{X}g(f(\mathbf{X})) = (\nabla_\mathbf{X}f)(\nabla_f g)$$

Quy tắc này cũng giống như trong hàm một biến

$$(g(f(x)))' = f'(x)g'(f)$$

Hãy lưu ý về sự phù hợp của kích thước ma trận khi làm việc với tích các ma trận.

## 2.4. Gradient của các hàm số thường gặp

### 2.4.1. $f(\mathbf{x}) = \mathbf{a^Tx}$

Giả sử $\mathbf{a, x} \in \mathbb{R}^n$, ta viết lại $f(\mathbf{x}) = \mathbf{a^Tx} = a_1x_1 + a_2x_2 + ... +a_nx_n$

Nhận thấy $\frac{\partial f(\mathbf{x})}{\partial x_i} = a_i,\ i = \overline{1,n}$

Vậy $\nabla_\mathbf{x}(\mathbf{a^Tx}) = [a_1, a_2, ..., a_n]^T = \mathbf{a}$

Ngoài ra, vì $\mathbf{a^Tx = x^Ta}$ nên $\nabla_\mathbf{x} (\mathbf{x^Ta}) = \mathbf{a}$

### 2.4.2. $f(\mathbf{x}) = \mathbf{Ax}$

Đây là một hàm trả về vector $f:\mathbb{R}^n \to \mathbb{R}^m$ với $\mathbf{x} \in \mathbb{R^n},\ \mathbf{A} \in \mathbb{R}^{m \times n}$. Giả sử $\mathbf{a_i}$ là hàng thứ $i$ của ma trận $\mathbf{A}$. Ta có

$$\mathbf{Ax} = \left[\begin{matrix}
  \mathbf{a}_1\mathbf{x} \\ \mathbf{a}_2\mathbf{x} \\ \vdots \\ \mathbf{a}_m\mathbf{x}
\end{matrix} \right]$$

Vậy từ định nghĩa gradient của ma trận và công thức gradient của $\mathbf{a}_i\mathbf{x}$, suy ra

$$\nabla_\mathbf{x} (\mathbf{Ax}) = [\mathbf{a_1^T, a_2^T, ..., a_m^T}] = \mathbf{A^T}$$

Từ đây suy ra đạo hàm của $f(\mathbf{x}) = \mathbf{x} = \mathbf{Ix}$ là

$$\nabla\mathbf{x} = I$$

### 2.4.3. $f(\mathbf{x}) = \mathbf{x^TAx}$

Với $\mathbf{x} \in \mathbb{R}^n,\ \mathbf{A} \in \mathbb{R}^{n \times n}$, áp dụng quy tắc tích ta có:

$$\begin{align*}
  \nabla f(\mathbf{x}) &= \nabla((\mathbf{x^T})(\mathbf{Ax})) \\
                      &= (\nabla(\mathbf{x}))\mathbf{Ax} + (\nabla (\mathbf{Ax}))\mathbf{x}\\
                      &= \mathbf{IAx} + \mathbf{A^Tx} \\
                      &= (\mathbf{A + A^T})\mathbf{x}
\end{align*}$$

Từ đó có thể suy ra $\nabla^2 \mathbf{x^TAx} = (\mathbf{A + A^T})$. Nếu $\mathbf{A}$ là ma trận đối xứng thì ta có $\nabla \mathbf{x^TAx = 2Ax},\ \nabla^2\mathbf{x^TAx} = 2\mathbf{A}$.

### 2.4.4. $f(\mathbf{x}) = \|\mathbf{Ax - b}\|_2^2$

Có hai cách để triển khai gradient của hàm số này:

- Cách 1: Trước hết khai triển:

$$f(\mathbf{x}) = \|\mathbf{Ax - b}\|_2^2 = (\mathbf{Ax - b})^T(\mathbf{Ax - b}) = \mathbf{(x^TA^T - b^T)(Ax - b)} = \mathbf{x^TA^TAx -2b^TAx + b^Tb}$$

Lấy gradient cho từng số hạng rồi ta có:

$$\nabla \|\mathbf{Ax - b}\|_2^2 = \mathbf{2A^TAx - 2A^Tb} = 2\mathbf{A^T(Ax - b)}$$

- Cách 2: Sử dụng $\nabla(\mathbf{Ax - b}) = \mathbf{A^T}$ và $\nabla\mathbf{\|x\|_2^2} = 2\mathbf{x}$ và quy tắc chuỗi ta thu được kết quả tương tự.

### 2.4.5. $f(\mathbf{x}) = \mathbf{a^Txx^Tb}$

Viết lại $f(\mathbf{x}) = \mathbf{(a^Tx)(x^Tb)}$ và dùng quy tắc tích, ta có:

$$\nabla(\mathbf{a^TxxTb}) = \mathbf{ax^Tb + b^Ta^Tx} = \mathbf{ab^Tx + ba^Tx = (ab^T + ba^T)x}$$

### 2.4.6. $f(\mathbf{X}) = trace(\mathbf{AX})$

Giả sử $\mathbf{A} \in \mathbb{R}^{n \times m},\ \mathbf{X} \in \mathbb{R}^{m \times n}$ và $\mathbf{B} = AX \in \mathbb{R}^{n \times n}$. Theo định nghĩa của trace:

$$f(\mathbf{X}) = trace (\mathbf{AX}) = trace(\mathbf{B}) = \sum_{i=1}^{n} b_ii = \sum_{j=1}^n \sum{i=1}^n a_{ji} x{ji}$$

Từ đó suy ra $\displaystyle\frac{\partial f(\mathbf{X})}{\partial x_{ij}} = a_{ji}$. Vậy theo định nghĩa gradient của ma trận ta có $trace(\mathbf{AX}) = \mathbf{A^T}$

### 2.4.7. $f(\mathbf{X}) = \mathbf{a^TXb}$

Giả sử rằng $\mathbf{a} \in \mathbb{R}^m,\ \mathbf{X} \in \mathbb{R}^{m \times n}$ và $\mathbf{b} \in \mathbb{R}^n$. Ta có thể chứng minh được:

$$f(\mathbf{X}) = \sum_{i=1}^m \sum_{j=1}^n x_{ij}a_ib_j$$

Từ đó sử dụng định nghĩa của phép gradient ma trận ta có

$$\nabla_{\mathbf{X}}\left(\mathbf{a}^T \mathbf{X} \mathbf{b}^T\right)=\left[\begin{array}{cccc}a_1 b_1 & a_1 b_2 & \ldots & a_1 b_n \\ a_2 b_1 & a_2 b_2 & \ldots & a_2 b_n \\ \ldots & \ldots & \ddots & \ldots \\ \ldots & \ldots & \ldots \\ a_m b_1 & a_m b_2 & \ldots & a_m b_n\end{array}\right]=\mathbf{a b}^T$$

### 2.4.8. $f(\mathbf{X}) = \|\mathbf{X}\|_F^2$

Giả sử $\mathbf{X} \in \mathbb{R}^{n\times n}$, ta có:

$$\|\mathbf{X}\|_F^2 = \sum_{i=1}^m \sum_{j=1}^n x_{ij}^2 \implies \frac{\partial f}{\partial x_{ij}} = 2x_{ij} \implies \nabla \|\mathbf{X}\|_F^2 = 2\mathbf{X}$$

### 2.4.9. $f(\mathbf{X}) = trace(\mathbf{X^TAX})$

Giả sử rằng $\mathbf{X=[x_1,x_2,...,x_n]} \in \mathbb{R^{m \times n}}$. Bằng cách khai triển:

$$\mathbf{X}^T\mathbf{A}\mathbf{X}=\begin{bmatrix}\mathbf{x}_1^T\\\mathbf{x}_2^T\\\vdots\\\mathbf{x}_n^T\end{bmatrix}\mathbf{A}\begin{bmatrix}\mathbf{x}_1 \mathbf{x}_2 \ldots \mathbf{x}_n\end{bmatrix}=\begin{bmatrix}\mathbf{x}_1^T\mathbf{A}\mathbf{x}_1 \mathbf{x}_1^T\mathbf{A}\mathbf{x}_2 \ldots \mathbf{x}_1^T\mathbf{A}\mathbf{x}_n\\\mathbf{x}_2^T\mathbf{A}\mathbf{x}_1 \mathbf{x}_2^T\mathbf{A}\mathbf{x}_2 \ldots \mathbf{x}_2^T\mathbf{A}\mathbf{x}_n\\\cdots \cdots \ddots \cdots\\\mathbf{x}_n^T\mathbf{A}\mathbf{x}_1 \mathbf{x}_n^T\mathbf{A}\mathbf{x}_2 \ldots \mathbf{x}_n^T\mathbf{A}\mathbf{x}_n\end{bmatrix}$$

Ta tính được $trace(\mathbf{X^TAX}) = \sum_{i=1}^n \mathbf{x_i^TAx_i}$. Sau đó ta sử dụng công thức

$$\nabla_\mathbf{X}\mathrm{trace}(\mathbf{X}^T\mathbf{A}\mathbf{X})=(\mathbf{A}+\mathbf{A}^T)\begin{bmatrix}\mathbf{x}_1 \mathbf{x}_2 \ldots \mathbf{x}_n\end{bmatrix}=(\mathbf{A}+\mathbf{A}^T)\mathbf{X}.$$

### 2.4.10. $f(\mathbf{X}) = \|\mathbf{AX - B}\|_F^2$

Tương tự 2.4.4 ta thu được

$$\nabla_\mathbf{X} \|\mathbf{AX - B}\|_F^2 = 2\mathbf{A^T(AX - B)}$$

## 2.5. Bảng các gradient thường gặp

| $f(\mathbf{x})$ | $\nabla f(\mathbf{x})$ | $f(\mathbf{X})$ | $\nabla_\mathbf{X} f(\mathbf{X})$ |
|---|---|---|---|
| $\mathbf{x}$ | b | c | d |
| $\mathbf{a^Tx}$ | b | c | d |
| $\mathbf{x^TAx}$ | b | c | d |
| $\mathbf{x^Tx =\|\|x\|\|_2^2}$ | b | c | d |
| $\mathbf{\|\|Ax-b\|_2^2}\|\|$ | b | c | d |
| $\mathbf{a}^T(\mathbf{x}^T\mathbf{x})\mathbf{b}$ | b | c | d |
| $\mathrm{a}^T\mathrm{x}\mathrm{x}^T\mathrm{b}$ | b | c | d |


## 2.6. Kiểm tra gradient

Việc tính gradient của hàm nhiều biến có thể phức tạp và dễ mắc lỗi. Có cách để kiểm tra liệu gradient tính được có chính xác không. Cách này dựa trên định nghĩa của đạo hàm hàm một biến.

### 2.6.1. Xấp xĩ đạo hàm của hàm một biến

Xét cách tính đạo hàm của hàm một biến theo định nghĩa:

$$f^{\prime}(x)=\lim _{\varepsilon \rightarrow 0} \frac{f(x+\varepsilon)-f(x)}{\varepsilon}$$

Trên mày tính, ta có thể chọn $\varepsilon$ rất nhỏ, rồi xấp xĩ đạo hàm này bởi

$$f'(x)\approx\lim_{\varepsilon\to0}\frac{f(x+\varepsilon)-f(x)}\varepsilon.$$

Trên thực tế, công thức xấp xĩ đạo hàm hai phía thường được sử dụng:

$$f'(x)\approx\frac{f(x+\varepsilon)-f(x-\varepsilon)}{2\varepsilon}.$$(2.20)

Cách tính này gọi là *numerical gradient*. Có hai cách giải thích vì sao cách tính như 2.20 rộng rãi:

- Cách 1: Bằng giải tích:

Sử dụng khai triển Taylor với $\varepsilon$ rẩt nhỏ, ta có hai xấp xĩ sau:

$$f(x+\varepsilon)\approx f(x)+f^{\prime}(x)\varepsilon+\frac{f"(x)}2\varepsilon^2+\frac{f^{(3)}}6\varepsilon^3+\ldots $$(2.21)

$$\frac{f(x+\varepsilon)-f(x-\varepsilon)}{2\varepsilon}\approx f'(x)+\frac{f'^{(3)}(x)}6\varepsilon^2+\cdots=f'(x)+O(\varepsilon^2).$$(2.22)

Trong đó $O()$ là Big O notation.

Từ đó, nếu xấp xỉ đạo hàm bằng công thức (2.23), sai số sẽ là $O(\varepsilon)$. Trong khi đó, nếu xấp xĩ đạo hàm bằng coogn thức (2.24), sai số là $O(\varepsilon^2)$. Khi $\varepsilon$ rất nhỏ thì: $O(\varepsilon^2)\ll O(\varepsilon),$ tức cách sử dụng (2.24) hiệu quả hơn, sai số nhỏ hơn.

### 2.6.2. Xấp xĩ gradient của hàm nhiều biến

Với hàm nhiều biến công thức (2.24) được áp dụng cho từng biến khi các biến khác cố định. Cụ thể, ta sử dụng định nghĩa gradient của một hàm số nhận đầu vào là một ma trận. Mỗi thành phần của ma trận kết quả là đạo hàm riêng của hàm số theo thành phần đó.

Cách tính gradient xấp xĩ hai phía thường cho giá trị khá chính xác. Tuy nhiên, cách này không được sử dụng để tính gradient vì độ phức tạp quá cao so với tính trực tiếp. Tại mỗi thành phần, ta cần tính giá trị của hàm số tại phía trái và phía phải. Việc làm này không khả thi với ma trận lớn. Khi so sánh đạo hàm xấp xĩ với gradient tính theo công thức, người ta thường giảm số chiều dữ liệu và giảm số điểm dữ liệu để thuận tiện cho việc tính toán. Nếu gradient tính được là chính xác, nó sẽ rất gần với giá trị xấp xĩ này.

Đoạn code dưới đây giúp kiểm tra gradient của một hàm số khả vi $f:\mathbb{R}^{m \times n} \to \mathbb{R}$, có kèm theo ví dụ. Để sử dụng hàm `check_grad`, ta cần hai hàm: `fn(X)` để tính giá trị hàm số tại $\mathbf{X}$. Hàm thứ hai là `gr(X)` để tính gradient của `fn(X)`.

```python
import numpy as np

np.random.seed(42)

def check_grad(fn, gr, X):
    X_flat = X.reshape(-1) # chuyển ma trận X về mảng 1 chiều
    shape_X = X.shape # lưu lại shape của X
    num_grad = np.zeros_like(X) # tạo ma trận numerical grad có shape giống X
    grad_flat = np.zeros_like(X_flat) # tạo mảng 1 chiều grad_flat có shape giống X_flat
    eps = 1e-6 # giá trị epsilon
    numElems = X_flat.shape[0] # số phần tử của X_flat
    # tính toán numerical gradient
    for i in range(numElems): # duyệt qua tất cả phần tử của X
        Xp_flat = X_flat.copy() 
        Xn_flat = X_flat.copy()
        Xp_flat[i] += eps
        Xn_flat[i] -= eps
        Xp = Xp_flat.reshape(shape_X)
        Xn = Xn_flat.reshape(shape_X)
        grad_flat[i] = (fn(Xp) - fn(Xn)) / (2 * eps)
    num_grad = grad_flat.reshape(shape_X)
    diff = np.linalg.norm(num_grad-gr(X)) # tính toán độ lệch giữa numerical gradient và gradient thực
    print('Difference between two methods should be small:', diff)

# Hàm số cần tính gradient: grad(trace(A*X)) == A^T

m, n = 10, 20
A = np.random.randn(m, n)
X = np.random.randn(n, m)

def fn1(X):
    return np.trace(A.dot(X))

def gr1(X):
    return A.T

check_grad(fn1, gr1, X)

# Hàm số cần tính gradient: grad(x^T*A*x) == (A + A^T)*x

A = np.random.randn(m, m)
x = np.random.rand(m, 1)

def fn2(x):
    return x.T.dot(A).dot(x)

def gr2(x):
    return (A + A.T).dot(x)

check_grad(fn2, gr2, x)

```


















