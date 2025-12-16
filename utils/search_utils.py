from elasticsearch import Elasticsearch

es = Elasticsearch(
    ['http://localhost:9200'],
    # 在做任何操作之前，先进行嗅探
    # sniff_on_start=True,
    # # 节点没有响应时，进行刷新，重新连接
    sniff_on_connection_fail=True,
    # # 每 60 秒刷新一次
    sniffer_timeout=240
)

def delete_index(index_name, delete_film):
    if delete_film is None:
        delete_film = []
    index_name = index_name
    es = Elasticsearch(
        ['http://localhost:9200'],
        # 在做任何操作之前，先进行嗅探
        # sniff_on_start=True,
        # # 节点没有响应时，进行刷新，重新连接
        sniff_on_connection_fail=True,
        # # 每 60 秒刷新一次
        sniffer_timeout=60
    )
    if es.indices.exists(index_name):
        # 删除指定电影序号的索引
        if delete_film:
            es.delete_by_query(index=index_name, body={'query':
                {'range': {
                    'film': {'gte': delete_film[0], 'lte': delete_film[1]}}}
            })
        # 删除所有电影的索引
        else:
            es.indices.delete(index=index_name)
        return True
    else:
        return False


def add_document(npy, bbox, feature_name):
    name = {'0': 'landmark', '1': 'people', '2': 'face', '3': 'clothing', '4': 'vehicle', '5': 'brand', '6': 'pedestrian'}
    index_name = name[feature_name.split('_')[-2]]


    # 判断索引是否存在
    if es.indices.exists(index_name):
        # 为索引添加文档
        # 检测后的结果是两点坐标，需要转化为左顶点与宽高
        bbox_old = bbox
        bbox[0], bbox[1], bbox[2], bbox[3] = bbox_old[0], bbox_old[1], bbox_old[2] - bbox_old[0], bbox_old[3] - \
                                             bbox_old[1]
        doc_vector = npy
        info = feature_name.split('_')
        id = int(info[0] + info[1] + info[2] + info[3])
        es.index(index=index_name,
                 body={
                     'features': doc_vector,
                     'image_name': feature_name,
                     'bbox': bbox,
                     'film': info[0]
                 },
                 id=id)

    else:
        print('索引不存在，需要创建！')
        request_body = {
            "mappings": {
                "properties": {
                    "features": {
                        "type": "dense_vector",
                        "dims": len(npy)
                    },
                    "image_name": {
                        "type": "keyword"
                    },
                    "bbox": {
                        "type": "integer"
                    },
                    "film": {
                        "type": "integer"
                    }
                }
            }
        }
        es.indices.create(index=index_name, body=request_body)
        print(index_name + "索引创建完成！")


def search(vector, index_name, threshold_bool, threshold):
    retrieval_size = 10000 # 结果数
    if not threshold_bool or index_name in ['clothing', 'brand']:
        threshold = 0
        retrieval_size = 10
    index_name = index_name
    search_result = ''
    result_bbox = []

    es = Elasticsearch(
        ['http://localhost:9200'],
        # 在做任何操作之前，先进行嗅探
        # sniff_on_start=True,
        # # 节点没有响应时，进行刷新，重新连接
        sniff_on_connection_fail=True,
        # # 每 60 秒刷新一次
        sniffer_timeout=60,
        timeout = 180
    )

    if es.indices.exists(index_name):
        # 根据向量query_vector进行查询
        # query_vector = np.load(path + 'test.npy')
        query_vector = vector
        # 编辑查询体
        test_body = {
            "query": {

                "script_score": {
                    "query": {
                        "match_all": {},
                        # "range": {'film': {'gt': 40}}
                        # "match":{
                        #     "image_name": '7_136_4_0'
                        # }
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vector,'features')+1",
                        "params": {
                            "query_vector": query_vector
                        }
                    }
                }
            }
        }

        query = es.search(index=index_name, body=test_body, scroll='5m', size=retrieval_size)
        results = query['hits']['hits']  # es查询出的结果第一页
        result_score = []
        for result in results:
            source = result['_source']
            # print(result['_score'] - 1)
            if result['_score'] - 1 > threshold:
                for key, value in source.items():
                    if key == 'image_name':
                        search_result = search_result + ' ' + value
                    if key == 'bbox':
                        result_bbox.append(value)
                result_score.append(result['_score'] - 1)
        search_result = search_result.strip(' ')
        return search_result, result_bbox, result_score
