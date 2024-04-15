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
    - [2.4.5. $f(\\mathbf{X}) = trace(\\mathbf{AX})$](#245-fmathbfx--tracemathbfax)


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

### 2.4.5. $f(\mathbf{X}) = trace(\mathbf{AX})$


















