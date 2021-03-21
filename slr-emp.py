
import pandas as pd   
import numpy as np 

emp = pd.read_csv("C:/Users/user/desktop/datasets/emp_data.csv")
emp=emp.rename(columns={"Churn_out_rate":"chrate"})
emp=emp.rename(columns={"Salary_hike":"salhike"})

import matplotlib.pylab as plt 

plt.scatter(x=emp['salhike'],y=emp['chrate'])
np.corrcoef(x=emp['salhike'],y=emp['chrate'])

#linear transformation

import statsmodels.formula.api as smf
model = smf.ols('chrate~salhike',data=emp).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(emp['salhike']))
pred1
print (model.conf_int(0.05)) 

res = emp.chrate - pred1
sqres = res*res
mse = np.mean(sqres)
rmse = np.sqrt(mse)
rmse



# Log Transformation

plt.scatter(x=np.log(emp['salhike']),y=emp['chrate'],color='yellow')
np.corrcoef(np.log(emp.salhike), emp.chrate)

model2 = smf.ols('chrate ~ np.log(salhike)',data=emp).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(emp['salhike']))
pred2
print(model2.conf_int(0.05))

res2 = emp.chrate - pred2
sqres2 = res2*res2
mse2 = np.mean(sqres2)
rmse2 = np.sqrt(mse2)
rmse2
# Exponential transformation
plt.scatter(x=emp['salhike'], y=np.log(emp['chrate']),color='orange')

np.corrcoef(emp.salhike, np.log(emp.chrate)) 

model3 = smf.ols('np.log(chrate) ~ salhike',data=emp).fit()
model3.summary()

pred_log = model3.predict(pd.DataFrame(emp['salhike']))
pred_log
pred3 = np.exp(pred_log)
pred3
print(model3.conf_int(0.05))

res3 = emp.chrate - pred3
sqres3 = res3*res3
mse3 = np.mean(sqres3)
rmse3 = np.sqrt(mse3)
rmse3
#ape
model4 = smf.ols('np.log(chrate)~salhike+I(salhike*salhike)',data=emp).fit()
model4.summary()

pred_log2 = model4.predict(pd.DataFrame(emp['salhike']))
pred_log2
pred4 = np.exp(pred_log2)
pred4
print (model4.conf_int(0.05))
res4 = emp.chrate - pred4
sqres4 = res4*res4
mse4 = np.mean(sqres4)
rmse4 = np.sqrt(mse4)
rmse4
