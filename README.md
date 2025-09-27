---
jupyter:
  kernelspec:
    display_name: "Python \\[conda env:base\\] \\*"
    language: python
    name: conda-base-py
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.12.7
  nbformat: 4
  nbformat_minor: 4
---

::: {.cell .markdown}
# Log Transformations - Lab
:::

::: {.cell .markdown}
## Introduction

It\'s time to practice some logarithmic transformations on the Ames
Housing dataset!
:::

::: {.cell .markdown}
## Objectives

You will be able to:

-   Determine if a log transformation would be useful for a specific
    model or set of data
-   Apply log transformations to independent and dependent variables in
    linear regression
-   Interpret the coefficients of variables that have been transformed
    using a log transformation
:::

::: {.cell .markdown}
## Ames Housing Data

Below we load the numeric features from the Ames Housing dataset into a
dataframe. We also drop any rows with missing data.
:::

::: {.cell .code execution_count="1"}
``` python
# Run this cell without changes
import pandas as pd
ames = pd.read_csv("ames.csv", index_col=0)
ames = ames.select_dtypes("number")
ames.dropna(inplace=True)
ames
```

::: {.output .execute_result execution_count="1"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinSF2</th>
      <th>...</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SalePrice</th>
    </tr>
    <tr>
      <th>Id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>60</td>
      <td>65.0</td>
      <td>8450</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>2003</td>
      <td>196.0</td>
      <td>706</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>61</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
      <td>80.0</td>
      <td>9600</td>
      <td>6</td>
      <td>8</td>
      <td>1976</td>
      <td>1976</td>
      <td>0.0</td>
      <td>978</td>
      <td>0</td>
      <td>...</td>
      <td>298</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>60</td>
      <td>68.0</td>
      <td>11250</td>
      <td>7</td>
      <td>5</td>
      <td>2001</td>
      <td>2002</td>
      <td>162.0</td>
      <td>486</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>4</th>
      <td>70</td>
      <td>60.0</td>
      <td>9550</td>
      <td>7</td>
      <td>5</td>
      <td>1915</td>
      <td>1970</td>
      <td>0.0</td>
      <td>216</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>35</td>
      <td>272</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>60</td>
      <td>84.0</td>
      <td>14260</td>
      <td>8</td>
      <td>5</td>
      <td>2000</td>
      <td>2000</td>
      <td>350.0</td>
      <td>655</td>
      <td>0</td>
      <td>...</td>
      <td>192</td>
      <td>84</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>250000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1456</th>
      <td>60</td>
      <td>62.0</td>
      <td>7917</td>
      <td>6</td>
      <td>5</td>
      <td>1999</td>
      <td>2000</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>2007</td>
      <td>175000</td>
    </tr>
    <tr>
      <th>1457</th>
      <td>20</td>
      <td>85.0</td>
      <td>13175</td>
      <td>6</td>
      <td>6</td>
      <td>1978</td>
      <td>1988</td>
      <td>119.0</td>
      <td>790</td>
      <td>163</td>
      <td>...</td>
      <td>349</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2010</td>
      <td>210000</td>
    </tr>
    <tr>
      <th>1458</th>
      <td>70</td>
      <td>66.0</td>
      <td>9042</td>
      <td>7</td>
      <td>9</td>
      <td>1941</td>
      <td>2006</td>
      <td>0.0</td>
      <td>275</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>60</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2500</td>
      <td>5</td>
      <td>2010</td>
      <td>266500</td>
    </tr>
    <tr>
      <th>1459</th>
      <td>20</td>
      <td>68.0</td>
      <td>9717</td>
      <td>5</td>
      <td>6</td>
      <td>1950</td>
      <td>1996</td>
      <td>0.0</td>
      <td>49</td>
      <td>1029</td>
      <td>...</td>
      <td>366</td>
      <td>0</td>
      <td>112</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>2010</td>
      <td>142125</td>
    </tr>
    <tr>
      <th>1460</th>
      <td>20</td>
      <td>75.0</td>
      <td>9937</td>
      <td>5</td>
      <td>6</td>
      <td>1965</td>
      <td>1965</td>
      <td>0.0</td>
      <td>830</td>
      <td>290</td>
      <td>...</td>
      <td>736</td>
      <td>68</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>2008</td>
      <td>147500</td>
    </tr>
  </tbody>
</table>
<p>1121 rows × 37 columns</p>
</div>
```
:::
:::

::: {.cell .markdown}
## Identify Good Candidates for Log Transformation

Below we plot each of the potential numeric features against
`SalePrice`:
:::

::: {.cell .code execution_count="2"}
``` python
# Run this cell without changes
import matplotlib.pyplot as plt
import numpy as np

y = ames["SalePrice"]
X = ames.drop("SalePrice", axis=1)

fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(15,15), sharey=True)

for i, column in enumerate(X.columns):
    # Locate applicable axes
    row = i // 6
    col = i % 6
    ax = axes[row][col]
    
    # Plot feature vs. y and label axes
    ax.scatter(X[column], y, alpha=0.2)
    ax.set_xlabel(column)
    if col == 0:
        ax.set_ylabel("SalePrice")

fig.tight_layout()
```

::: {.output .display_data}
![](4b489cf77293293142530a07cad95ba86afe2aee.png)
:::
:::

::: {.cell .markdown}
Let\'s say we want to build a model with **at least one log-transformed
feature** as well as a **log-transformed target**

Do you see any features that look like good candidates for this type of
transformation?

For reference, a good candidate for this might look like any of these
three graphs:

------------------------------------------------------------------------

```{=html}
<div align="center"><div style="background-image: url('https://upload.wikimedia.org/wikipedia/commons/thumb/0/00/Population_vs_area.svg/256px-Population_vs_area.svg.png'); height: 200px; width: 256px;"></div><a title="Skbkekas, CC BY-SA 3.0 &lt;http://creativecommons.org/licenses/by-sa/3.0/&gt;, via Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File:Population_vs_area.svg">Skbkekas, CC BY-SA 3.0, via Wikimedia Commons</a></div>
```

------------------------------------------------------------------------

```{=html}
<div align="center"><img src="http://sciences.usca.edu/biology/zelmer/305/trans/y.jpg" width="256"/>
<a href="http://sciences.usca.edu/biology/zelmer/305/trans/">Derek Zelmer, UCSA</a></div>
```

------------------------------------------------------------------------

![e\^x](https://curriculum-content.s3.amazonaws.com/data-science/images/log_graphs.png)
:::

::: {.cell .markdown}
Try to find one feature that resembles each of these shapes.

Because this is real-world messy data, none of them are going to match
perfectly, and that\'s ok!
:::

::: {.cell .code execution_count="3"}
``` python
# Your written answer here
ames['GrLivArea'], ames['GarageArea'], ames['1stFlrSF']
```

::: {.output .execute_result execution_count="3"}
    Id
    1       1710
    2       1262
    3       1786
    4       1717
    5       2198
            ... 
    1456    1647
    1457    2073
    1458    2340
    1459    1078
    1460    1256
    Name: GrLivArea, Length: 1121, dtype: int64
:::
:::

::: {.cell .markdown}
### Plot Log Transformed Versions of Features

For each feature that you identified as a good candidate for log
transformation, plot the feature vs. `SalePrice` as well as the log
transformed feature vs. log transformed `SalePrice`.
:::

::: {.cell .code execution_count="9"}
``` python
# Your code here
X = [ames['GrLivArea'], ames['GarageArea'], ames['1stFlrSF']]
y = ames['SalePrice']

for x in X:
    plt.scatter(x, y)
    plt.show()
    
```

::: {.output .display_data}
![](84388309fe0a620c0c6f70153aeae789b3de8074.png)
:::

::: {.output .display_data}
![](860fc43a37969dca653178a694c38be890133c7f.png)
:::

::: {.output .display_data}
![](627400c3d62185a5b85e8f329ebbef4019f8f36d.png)
:::
:::

::: {.cell .code execution_count="29"}
``` python
# log transformations
# Convert to numeric
for col in ['GrLivArea', 'GarageArea', '1stFlrSF', 'SalePrice']:
    ames[col] = pd.to_numeric(ames[col], errors='coerce')

cols_to_transform = ['GrLivArea', 'GarageArea', '1stFlrSF', 'SalePrice']
ames[cols_to_transform] = ames[cols_to_transform].apply(np.log1p)


for col in cols_to_transform:
    plt.scatter(ames[col], ames['SalePrice'])
    plt.show()
```

::: {.output .display_data}
![](3cadf7ff689afa4fb9b82d11b88482f0766aa0b7.png)
:::

::: {.output .display_data}
![](6466f7bc6c09bcf222a56882610f058f3a93cf15.png)
:::

::: {.output .display_data}
![](aeaccc77102e2163bc8cc9f628d6fc7fe90b6048.png)
:::

::: {.output .display_data}
![](f14168b306bba82987b8fdf3a912a0002d6bd624.png)
:::
:::

::: {.cell .markdown}
Do the transformed relationships look more linear? If so, they should be
included in the model.
:::

::: {.cell .markdown}
## Build a Model with Log-Transformed Features and Target

### Data Preparation

Choose up to 3 of the features you investigated, and set up an X
dataframe containing the log-transformed versions of these features as
well as a y series containing the log-transformed version of the target.

------------------------------------------------------------------------

```{=html}
<details>
    <summary style="cursor: pointer"><b>Hint (click to reveal)</b></summary>

If you are planning log transform a feature measured in _years_ (e.g. `YearRemodAdd`) consider shifting the data first. For example, you might subtract 1900 or 1910 from the year, so that a 1% increase in year is closer to meaning 1 year rather than 20 years.

</details>
```
:::

::: {.cell .code execution_count="41"}
``` python
# Your code here - prepare data for modeling
y_log = ames['SalePrice']
X_log = ames[['GrLivArea', 'GarageArea', '1stFlrSF']]

```
:::

::: {.cell .code execution_count="38"}
``` python
for x in X_log:
    print(ames[x])
    print(ames[x].dtypes)
```

::: {.output .stream .stdout}
          GrLivArea  GarageArea  1stFlrSF
    Id                                   
    1      0.528780    0.476610  0.498980
    2      0.516343    0.467000  0.516343
    3      0.530486    0.482080  0.502341
    4      0.528941    0.484879  0.504343
    5      0.538391    0.497863  0.512155
    ...         ...         ...       ...
    1456   0.527293    0.467000  0.503961
    1457   0.536200    0.471641  0.536200
    1458   0.540702    0.429535  0.513754
    1459   0.509506    0.426147  0.509506
    1460   0.516140    0.435697  0.516140

    [1121 rows x 3 columns]
    GrLivArea     float64
    GarageArea    float64
    1stFlrSF      float64
    dtype: object
:::
:::

::: {.cell .code execution_count="33"}
``` python
y_log
```

::: {.output .execute_result execution_count="33"}
    Id
    1       1.254287
    2       1.251033
    3       1.255899
    4       1.244811
    5       1.258475
              ...   
    1456    1.250169
    1457    1.254454
    1458    1.259931
    1459    1.245176
    1460    1.246075
    Name: SalePrice, Length: 1121, dtype: float64
:::
:::

::: {.cell .markdown}
### Modeling

Now build a StatsModels OLS model with a log-transformed target as well
as log-transformed features.
:::

::: {.cell .code execution_count="43"}
``` python
# Your code here - build a model
import statsmodels.api as sm

model = sm.OLS(y_log, sm.add_constant(X_log))
results = model.fit()
```
:::

::: {.cell .markdown}
### Model Evaluation and Interpretation

How did the model perform? How might we interpret its coefficients?
Create as many cells as needed.
:::

::: {.cell .code execution_count="44"}
``` python
# Your code here - evaluate the model
results.summary()
```

::: {.output .execute_result execution_count="44"}
```{=html}
<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>SalePrice</td>    <th>  R-squared:         </th> <td>   0.650</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.649</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   691.3</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 17 Sep 2025</td> <th>  Prob (F-statistic):</th> <td>5.49e-254</td>
</tr>
<tr>
  <th>Time:</th>                 <td>15:54:42</td>     <th>  Log-Likelihood:    </th> <td>  4230.1</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  1121</td>      <th>  AIC:               </th> <td>  -8452.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  1117</td>      <th>  BIC:               </th> <td>  -8432.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>      <td>    0.9438</td> <td>    0.007</td> <td>  130.236</td> <td> 0.000</td> <td>    0.930</td> <td>    0.958</td>
</tr>
<tr>
  <th>GrLivArea</th>  <td>    0.3376</td> <td>    0.015</td> <td>   22.202</td> <td> 0.000</td> <td>    0.308</td> <td>    0.367</td>
</tr>
<tr>
  <th>GarageArea</th> <td>    0.1193</td> <td>    0.009</td> <td>   13.730</td> <td> 0.000</td> <td>    0.102</td> <td>    0.136</td>
</tr>
<tr>
  <th>1stFlrSF</th>   <td>    0.1446</td> <td>    0.015</td> <td>    9.789</td> <td> 0.000</td> <td>    0.116</td> <td>    0.174</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>249.770</td> <th>  Durbin-Watson:     </th> <td>   2.069</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td> 660.720</td> 
</tr>
<tr>
  <th>Skew:</th>          <td>-1.153</td>  <th>  Prob(JB):          </th> <td>3.36e-144</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 5.971</td>  <th>  Cond. No.          </th> <td>    141.</td> 
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```
:::
:::

::: {.cell .markdown}
-   Our model is statistically significant.
-   All coefficients are statistically significant.
-   For each increase of 1% in above-grade living area, we see an
    associated increase of about 0.7% in sale price.
:::

::: {.cell .markdown}
## Summary

Now you have practiced modeling with log transformations! This is a
subtle, messy process, so don\'t be discouraged if this was a tricky
lab.
:::
