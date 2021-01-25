from predict_execution_blog import *
from db.almaden import Sql
db = Sql("datacast2")
task_row = db.select('crawl_task','*','crawl_status="GF" and channel !="navershopping"')
## pred_confing
obj_pred_config = Pred_config()

##predict
for row in task_row:
    task_id = row['task_id']
    channel =row['channel']

    db.update_one('crawl_task', 'crawl_status', 'SI', 'task_id', task_id)
    obj_predict = Predict(obj_pred_config,task_id=task_id,keyword=row['keyword'],channel=row['channel'])
    obj_predict.predict()
    db.update_one('crawl_task','crawl_status','SF','task_id',task_id)