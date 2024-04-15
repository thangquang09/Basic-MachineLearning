# Giải tích ma trận


- [Giải tích ma trận](#giải-tích-ma-trận)
  - [2.1. Gradient của hàm trả về một số vô hướng](#21-gradient-của-hàm-trả-về-một-số-vô-hướng)
  - [2.2. Gradient của một hàm trả về vector](#22-gradient-của-một-hàm-trả-về-vector)
  - [2.3. Tính chất quan trọng](#23-tính-chất-quan-trọng)


Giả sử các gradient tồn tại trong chương!

## 2.1. Gradient của hàm trả về một số vô hướng

Gradient bậc nhất của một hàm số $f\mathbf{(x)}:\mathbb{R}^n \to \mathbb{R}$ theo $\mathbf{x}$, ký hiệu là $\nabla_\mathbf{x}f(\mathbf{x})$, được định nghĩa bởi:

$$\nabla_\mathbf{x}f(\mathbf{x})\triangleq\begin{bmatrix}\dfrac{\partial f(\mathbf{x})}{\partial x_1}\\\dfrac{\partial f(\mathbf{x})}{\partial x_2}\\\vdots\\\dfrac{\partial f(\mathbf{x})}{\partial x_n}\end{bmatrix}\in\mathbb{R}^n$$

trong đó $\displaystyle\frac{\partial f(\mathbf{x})}{\partial x_i}$ là đạo hàm riêng của hàm số theo thành phần thứ $i$ của $\mathbf{x}$. Đạo hàm này được tính khi tất cả các biến ngoài $\mathbf{x}$ là hằng số. Nếu không có biến nào khác thì $\nabla_\mathbf{x} f(\mathbf{x})$ được viết gọn lại là $\nabla f(\mathbf{x})$. Gradient của hàm số là một vector cùng chiều với vector đang được lấy gradient. Nếu vector đó ở dạng cột thì gradient cũng ở dạng cột.

Gradient bậc hai của hàm số trên còn được gọi là *Hesse* (Hessian) và được định nghĩa như sau:

$$\begin{aligned}\nabla^2f(\mathbf{x})&\triangleq\begin{bmatrix}\frac{\partial^2f(\mathbf{x})}{\partial x_1^2}&\frac{\partial^2f(\mathbf{x})}{\partial x_1\partial x_2}\cdots\frac{\partial^2f(\mathbf{x})}{\partial x_1\partial x_n}\\\frac{\partial^2f(\mathbf{x})}{\partial x_2\partial x_1}&\frac{\partial^2f(\mathbf{x})}{\partial x_2^2}\cdots\frac{\partial^2f(\mathbf{x})}{\partial x_2\partial x_n}\\\vdots&\vdots&\ddots&\vdots\\\frac{\partial^2f(\mathbf{x})}{\partial x_n\partial x_1}&\frac{\partial^2f(\mathbf{x})}{\partial x_n\partial x_2}\cdots\frac{\partial^2f(\mathbf{x})}{\partial x_n^2}\end{bmatrix}\in\mathbb{S}^n.\end{aligned}$$

Gradient của một hàm số $f(\mathbf{X}): \mathbb{R}^{n \times\ m} \to \mathbb{R}$ theo ma trận $\mathbf{X}$ định nghĩa là:

$$\nabla f(\mathbf{X})=\begin{bmatrix}\dfrac{\partial f(\mathbf{X})}{\partial x_{11}}&\dfrac{\partial f(\mathbf{X})}{\partial x_{12}}&\cdots\dfrac{\partial f(\mathbf{X})}{\partial x_{1m}}\\\dfrac{\partial f(\mathbf{X})}{\partial x_{21}}&\dfrac{\partial f(\mathbf{X})}{\partial x_{22}}&\cdots\dfrac{\partial f(\mathbf{X})}{\partial x_{2m}}\\\vdots&\vdots&\ddots&\vdots\\\dfrac{\partial f(\mathbf{X})}{\partial x_{n1}}&\dfrac{\partial f(\mathbf{X})}{\partial x_{n2}}&\cdots\dfrac{\partial f(\mathbf{X})}{\partial x_{nm}}\end{bmatrix}\in\mathbb{R}^{n\times m}.$$

>Vậy Gradient của hàm số $f: \mathbb{R}^{n \times m} \to \mathbb{R}$ là một ma trận trong $\mathbb{R}^{n \times m}$

Cụ thể để tính gradient của một hàm $f: \mathbb{R}^{n \times m} \to \mathbb{R}$, ta tính đạo hàm riêng của hàm số đó theo từng thành phần khác được giả sử là hằng số. Tiếp theo ta sắp xếp các đạo hàm riêng tính được theo đúng thứ tự trong ma trận.

*Ví dụ:* Xét hàm số $f: \mathbb{R}^2 \to \mathbb{R},\ f(\mathbf{x}) = x_1^2 + 2x_1x_2 + sin(x_1) + 2$. Gradient bậc 1 theo $\mathbf{x}$ của hàm số đó là:

$$\nabla f(\mathbf{x})=\begin{bmatrix}\dfrac{\partial f(\mathbf{x})}{\partial x_1}\\\dfrac{\partial f(\mathbf{x})}{\partial x_2}\end{bmatrix}=\begin{bmatrix}\\2x_1+2x_2+\cos(x_1)\\\\2x_1\end{bmatrix}$$

Gradient bậc 2 theo $\mathbf{x}$, hay Hesse là:

$$\nabla^2f(\mathbf{x})=\begin{bmatrix}\dfrac{\partial^2f(\mathbf{x})}{\partial x_1^2}&\dfrac{\partial f^2(\mathbf{x})}{\partial x_1\partial x_2}\\\dfrac{\partial^2f(\mathbf{x})}{\partial x_2\partial x_1}&\dfrac{\partial f^2(\mathbf{x})}{\partial x_2^2}\end{bmatrix}=\begin{bmatrix}\\2-\sin(x_1)&2\\\\2&0\end{bmatrix}$$

>Chú ý: Hesse luôn là ma trận đối xứng.

## 2.2. Gradient của một hàm trả về vector

Xét một hàm trả về vector với đầu vào là một số thực $v(x) : \mathbb{R} \to \mathbb{R}^n$:

$$v(x)=\begin{bmatrix}v_1(x)\\v_2(x)\\\vdots\\v_n(x)\end{bmatrix}.$$

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























