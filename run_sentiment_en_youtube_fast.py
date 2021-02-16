from predict_execution_en_review import *
from db.almaden import Sql
db = Sql("datacast2")

contents_row = db.select('crawling_status_youtube_view','*','contents_status="GF" and is_channel=1')
## pred_confing

##predict
for row in contents_row:
    keyword= row['keyword']
    contents_id = row['contents_id']
    n_reply_crawled = row['n_reply_crawled']

    if n_reply_crawled is not None and n_reply_crawled > 0:
        db.update_one('crawl_contents','crawl_status','SI','contents_id',contents_id)
        obj_predict = Predict(keyword=row['keyword'], channel='youtube', contents_id=contents_id)
        obj_predict.predict()
    else:
        task_ids = db.update_one('crawl_contents', 'crawl_status', 'SF', 'contents_id', contents_id)
        pass
    db = Sql("datacast2")
    db.update_one('crawl_contents', 'crawl_status', 'SF', 'contents_id', contents_id)
