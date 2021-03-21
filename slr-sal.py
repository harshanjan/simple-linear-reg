
import pandas as pd   
import numpy as np 

sal = pd.read_csv("C:/Users/user/desktop/datasets/salary_data.csv")
sal=sal.rename(columns={"YearsExperience":"yearsexp"})
sal=sal.rename(columns={"Salary":"salary"})
#above method is useful when only particular column name to be changed
#simple method sal.colums = "year","salary"
import matplotlib.pylab as plt 

plt.scatter(x=sal['yearsexp'],y=sal['salary'])
np.corrcoef(x=sal['yearsexp'],y=sal['salary'])

#linear transformation

import statsmodels.formula.api as smf
model = smf.ols('salary~yearsexp',data=sal).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(sal['yearsexp']))
pred1
print (model.conf_int(0.05)) 

res = sal.salary - pred1
sqres = res*res
mse = np.mean(sqres)
rmse = np.sqrt(mse)
rmse



# Log Transformation

plt.scatter(x=np.log(sal['yearsexp']),y=sal['salary'],color='yellow')
np.corrcoef(np.log(sal.yearsexp), sal.salary)

model2 = smf.ols('salary ~ np.log(yearsexp)',data=sal).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(sal['yearsexp']))
pred2
print(model2.conf_int(0.05))

res2 = sal.salary - pred2
sqres2 = res2*res2
mse2 = np.mean(sqres2)
rmse2 = np.sqrt(mse2)
rmse2
# Exponential transformation
plt.scatter(x=sal['yearsexp'], y=np.log(sal['salary']),color='orange')

np.corrcoef(sal.yearsexp, np.log(sal.salary)) 

model3 = smf.ols('np.log(salary) ~ yearsexp',data=sal).fit()
model3.summary()

pred_log = model3.predict(pd.DataFrame(sal['yearsexp']))
pred_log
pred3 = np.exp(pred_log)
pred3
print(model3.conf_int(0.05))

res3 = sal.salary - pred3
sqres3 = res3*res3
mse3 = np.mean(sqres3)
rmse3 = np.sqrt(mse3)
rmse3
#ape
model4 = smf.ols('np.log(salary)~yearsexp+I(yearsexp*yearsexp)',data=sal).fit()
model4.summary()

pred_log2 = model4.predict(pd.DataFrame(sal['yearsexp']))
pred_log2
pred4 = np.exp(pred_log2)
pred4
print (model4.conf_int(0.05))
res4 = sal.salary - pred4
sqres4 = res4*res4
mse4 = np.mean(sqres4)
rmse4 = np.sqrt(mse4)
rmse4
