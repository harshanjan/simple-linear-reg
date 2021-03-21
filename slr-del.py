import pandas as pd   
import numpy as np 

dev = pd.read_csv("C:/Users/user/desktop/datasets/delivery_time.csv")
dev=dev.rename(columns={"Sorting Time":"sortingtime"})
dev=dev.rename(columns={"Delivery Time":"deliverytime"})

import matplotlib.pylab as plt 

plt.scatter(x=dev['sortingtime'],y=dev['deliverytime'])
np.corrcoef(x=dev['sortingtime'],y=dev['deliverytime'])

#linear transformation

import statsmodels.formula.api as smf
model = smf.ols('deliverytime~sortingtime',data=dev).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(dev['sortingtime']))
pred1
print (model.conf_int(0.05)) 

res = dev.deliverytime - pred1
sqres = res*res
mse = np.mean(sqres)
rmse = np.sqrt(mse)
rmse



# Log Transformation

plt.scatter(x=np.log(dev['sortingtime']),y=dev['deliverytime'],color='brown')
np.corrcoef(np.log(dev.sortingtime), dev.deliverytime)

model2 = smf.ols('deliverytime ~ np.log(sortingtime)',data=dev).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(dev['sortingtime']))
pred2
print(model2.conf_int(0.05))

res2 = dev.deliverytime - pred2
sqres2 = res2*res2
mse2 = np.mean(sqres2)
rmse2 = np.sqrt(mse2)
rmse2
# Exponential transformation
plt.scatter(x=dev['sortingtime'], y=np.log(dev['deliverytime']),color='orange')

np.corrcoef(dev.sortingtime, np.log(dev.deliverytime)) 

model3 = smf.ols('np.log(deliverytime) ~ sortingtime',data=dev).fit()
model3.summary()

pred_log = model3.predict(pd.DataFrame(dev['sortingtime']))
pred_log
pred3 = np.exp(pred_log)
pred3
print(model3.conf_int(0.05))

res3 = dev.deliverytime - pred3
sqres3 = res3*res3
mse3 = np.mean(sqres3)
rmse3 = np.sqrt(mse3)
rmse3
#ape
model4 = smf.ols('np.log(deliverytime)~sortingtime+I(sortingtime*sortingtime)',data=dev).fit()
model4.summary()

pred_log2 = model4.predict(pd.DataFrame(dev['sortingtime']))
pred_log2
pred4 = np.exp(pred_log2)
pred4
print (model4.conf_int(0.05))
res4 = dev.deliverytime - pred4
sqres4 = res4*res4
mse4 = np.mean(sqres4)
rmse4 = np.sqrt(mse4)
rmse4
