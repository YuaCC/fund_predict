spider_workers = 6;
spider_running_workers = 0;
spider_workers_waittime = 500;
spider_buf =[];
spider_buf_max_length = 256;

load_jjjz=function (idx){
    if (idx<spider_fund_ids.length){
        if (spider_buf.length < spider_buf_max_length){
            jQuery.getJSON(
                `//api.fund.eastmoney.com/f10/lsjz?callback=?&fundCode=${spider_fund_ids[idx]}&pageIndex=1&pageSize=2000000000&startDate=&endDate=`,
                function (data){
                    spider_buf.push([spider_fund_ids[idx],data['Data']['LSJZList']])
                    console.log(`${spider_fund_ids[idx]} ok, ${data['TotalCount']} items in total`);
                    load_jjjz(idx+spider_workers);
                }
            )
        }else
            setTimeout(load_jjjz,spider_workers_waittime,idx);
    }else
        spider_running_workers-=1;
}
load_jjjz_multi_workers = function (){
    spider_running_workers = spider_workers;
    for(let i=0;i<spider_workers;++i){
        load_jjjz(i);
    }
}
get_jjjz = function(){
    let ret = spider_buf;
    spider_buf = [];
    return ret;
}