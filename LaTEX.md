

```

```

# LaTEX 

> - 저는 수식을 전개, 증명 할 때, 수학적 개념을 간단한 그림, 도형으로 도식화 할 때 사용합니다. 

개인적으로 LaTEX문법을 사용하면서 알게된 것을 설명해야겠습니다. 

 * **공백**
 
  LaTEX에서는 Space, Tab, blank 등과 심지어 줄바꿈 또한 모두 동일하게 "공백 하나, 스페이스"로 처리가 됩니다. 그래서 줄을 구분하고 싶을 때는 사이에 줄을 한 번 더 띄워야 합니다. LaTEX로 문석작업을 할 땐, 작업하면서 동시에 결과물을 볼수 없기 때문에, 이 점을 알고 있다. 나타내고자하는 결과물과 실제 결과물의 차이를 없애고자하는 노력을 1차적으로 없앨 수 있습니다. 

 * **특수문자** 
 
  몇 몇 기호들은 LaTEX에서 특별한 의미를 가지지 않고, 직접적인 타이핑으로 나타낼 수 없는 것들이 있습니다. 이는 알수없는 오류 처럼 보이기 때문에 한 번 봐두어야 합니다. 
  
 ** { # ,  $ ,  % ,  &,  _,  { ,  }  ,  \  } **
  
  등이 그것인데, 이것들을 표현 하기 위해서는 해당 문자 앞에 '\'를 더해주어야 합니다. 
  
  $$ \# \$ \% \\ \& \_ \{ \} $$
  > '/'백슬래쉬의 경우, '//' 부호는 줄바꿈에 사용 된다.

#### 일반적인 수식의 전개 

- 수식은 일반적으로 \$$  \\$$(달러 2개)로 구간을 만듭니다.
- 문장 속에서 수식을 쓸 때는 \$ 와  \$ 사이에 작성합니다. 

$$ \dfrac{1}{1+\exp{(-z)}}$$

Logits는 odds ratio $\dfrac{\theta}{1 - \theta} $에 로그를 씌운 것이고 $\log{(\dfrac{\theta}{1 - \theta})}$ 이를 역함수를 취한 것이 위의 로지스틱 함수 이다. 

$ 행렬 A의 \det(A)는 A를 고유분해한 \lambda_1, \lambda_2 ... \lambda_n을 모두 곱한 값이다. 이를 수식으로 나타내면 $

$$
\text{A} \in \mathbf{R}^{N \times N}
\\
\text{A} = \prod_{i=1}^N{\lambda_n}
$$
좀 복잡한 식도 이렇게 쓸 수 있습니다. 
$$
\begin{eqnarray}
\mathcal{N}(x \mid \mu, \Sigma) \rightarrow\\
&=& \dfrac{1}{(2\pi)^{D/2} |\Sigma|^{1/2}} \exp \left( -\dfrac{1}{2} (x-\mu)^T \Sigma^{-1} (x-\mu) \right) \\
\\&=& \dfrac{1}{(2\pi)^{D/2} |\Sigma|^{1/2}} \exp \left( -\dfrac{1}{2} (x-\mu)^T V \Lambda^{-1} V^T (x-\mu) \right) \\
\\&=&\dfrac{1}{(2\pi)^{D/2} |\Sigma|^{1/2}} \exp \left( -\dfrac{1}{2} (V^T(x-\mu))^T  \Lambda^{-1} (V^T (x-\mu)) \right) \\
\\&=& \dfrac{1}{(2\pi)^{D/2} |\Sigma|^{1/2}} \exp \left( -\dfrac{1}{2} x'^T  \Lambda^{-1} x' \right) \\
\end{eqnarray}
$$

#### 벡터와 매트릭스의 표현 

$$
A=\begin{bmatrix}
           a_{11} & \cdots & a_{1n} \\
           \vdots & \ddots & \vdots \\
           a_{n1} & \cdots & a_{nn}
          \end{bmatrix}
$$

길게 잔뜩 붙여서 하는 것도 복잡해 보이지만, 자세히보면 간단합니다. 
(물론 저도 시행착오가 많았지만..)

$$
\begin{eqnarray}
\hat{y} = 
\begin{bmatrix}
\hat{y}_1 \\
\hat{y}_2 \\
\vdots \\
\hat{y}_M \\
\end{bmatrix}
&=& 
\begin{bmatrix}
w_1 x_{1,1} + \cdots + w_N x_{1,N} \\
w_1 x_{2,1} + \cdots + w_N x_{2,N} \\
\vdots  \\
w_1 x_{M,1} + \cdots + w_N x_{M,N} \\
\end{bmatrix}
\\
&=& 
\begin{bmatrix}
x_{1,1} & x_{1,2} & \cdots & x_{1,N} \\
x_{2,1} & x_{2,2} & \cdots & x_{2,N} \\
\vdots  & \vdots  & \vdots & \vdots \\
x_{M,1} & x_{M,2} & \cdots & x_{M,N} \\
\end{bmatrix}
\begin{bmatrix}
w_1 \\ w_2 \\ \vdots \\ w_N
\end{bmatrix}
\\
&=& 
\begin{bmatrix}
x_1^T \\
x_2^T \\
\vdots \\
x_M^T \\
\end{bmatrix}
\begin{bmatrix}
w_1 \\ w_2 \\ \vdots \\ w_N
\end{bmatrix}
\\
&=& X w 
\end{eqnarray}
$$


#### 마무리 
일단 이정도만 이해하셨으면, 현재로서 충분하다고 생각합니다. 이렇게 모아둔다고 해서 보기 쉬운 문서가 되지는 않아서 아쉽습니다. LaTEX는 논문이나 책을 쓸 때도 많이 사용한다고 합니다. 그림을 그리고 도식화하고 그림 삽입하는 등 여러가지 다양한 기능이 존재합니다. 하지만 지금 단계에서는 벡터와 행렬을 표현하고, 기본적인 수식을 적는 방법만 알면 나머지는 그때 그때 검색하여 알아가면 됩니다. 저도 그때그때 잊어버린것이나 모르는것을 찾아 참고하여 씁니다. 위의 것을 참고하시면 이번 박사님 과제는 어렵지 않게 할 수 있을 것이고, 추후 공부한 것들을 정리할 때 계속 써보시면 손에 익어 금방 능숙해질 것 입니다.







```

```


```

```


```

```


```

```


```

```


```

```


```

```


```

```


```

```


```

```


```

```


```

```


```

```


```

```


```

```


```

```


```

```


```

```


```

```


```

```


```

```


```

```


```

```


```

```


```

```


```

```


```

```


```

```


```

```


```

```


```

```


```

```


```

```


```

```


```

```


```

```


```

```


```

```


```

```


```

```


```

```


```

```


```

```


```

```


```

```


```

```


```

```


```

```

##### 박사님 과제 정답 

* 문제, $\| x \|^2 - (x^T\text{v})^2$ 를 증명하라.



```
v = np.array([1, 1])
x1 = np.array([4, 1])
x2 = np.array([2, 2]) * 1.5
w0 = 5
plt.figure(figsize=(9,9))
plt.annotate('', xy=x1, xytext=(0,0), arrowprops=dict(facecolor='gray'))
plt.annotate('', xy=x2, xytext=(0,0), arrowprops=dict(facecolor='gray'))
plt.annotate('', xy=v, xytext=(0,0), arrowprops=dict(facecolor='red'))

plt.text(0, 0.8, "$v$", fontdict={"size": 18})
plt.text(4, 1, "$x$", fontdict={"size": 18})
plt.text(2, 2.5, "$x^{\Vert v}$", fontdict={"size": 18})

plt.plot([6, -1], [-1, 6])

plt.xticks(np.arange(-1, 8))
plt.yticks(np.arange(-1, 8))
plt.xlim(-1, 8)
plt.ylim(-1, 8)
plt.show()
```


![png](output_54_0.png)


$$\|\text{v}\| = 1 $$
$$
\| d \| = Distance\\
\| x \|^2 = \| d \|^2 + \| x^{\Vert v} \|^2\\
$$
따라서 우리가 찾고자하는 $\| d \|^2 = \| x \|^2 - \| x^{\Vert v} \|^2$

여기서 투영된 $x^{\Vert v}$의 길이는  
$$
\| x^{\Vert v} \| = \| x \| \cos{\theta}\\
\| x^{\Vert v} \| = \dfrac{x^Tv}{\| v \|}\\
\|\text{v}\| = 1\\
\| x^{\Vert v} \| = x^Tv
$$
이렇게 전개 해보면 $\| x^{\Vert v} \| = x^Tv$임을 알수 있다.
$$
\\
$$

$$
\| d \|^2 = \| x \|^2 - (x^Tv)^2\\
$$



```

```


```

```
