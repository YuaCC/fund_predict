import datetime
import os
import time

from selenium.webdriver import Chrome, ChromeOptions


class TianTianFundSpider:
    '''
    remember to call close() when finished
    '''

    def __init__(self, chrome_driver, headless=True):
        option = ChromeOptions()
        if headless:
            option.add_argument("headless")
        self.driver = Chrome(chrome_driver, chrome_options=option)
        self.dataset_folder = "./dataset/"
        self.update_log_file = "./updatelog.csv"
        self.update_log_file_tmp = "./updatelog.csv.tmp"
        self.sgzt_file = './sgzt.csv'
        self.sgzt_file_tmp = './sgzt.csv.tmp'
        self.max_fund_num = 0x7fffffff
        self.fund_ids = self.get_fund_ids()[:self.max_fund_num]
        if not os.path.exists(self.dataset_folder):
            os.mkdir(self.dataset_folder)

        self.update_log = {}
        self.load_update_log()

        self.sgzt = {}
        self.load_sgzt()

    def load_sgzt(self):
        if not os.path.exists(self.sgzt_file):
            if os.path.exists(self.sgzt_file_tmp):
                os.rename(self.sgzt_file_tmp, self.sgzt_file)
            else:
                f = open(self.sgzt_file, 'w')
                f.close()
        with open(self.sgzt_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                fund_id, state = line.strip().split(",")
                if state == "True" or state=="False":
                    self.sgzt[fund_id] = state
                else:
                    raise ValueError(f"state {state} not supported yet")

    def save_sgzt(self):
        with open(self.sgzt_file_tmp, "w") as f:
            for fund_id in self.sgzt:
                if self.sgzt[fund_id] is not None:
                    f.write(f"{fund_id},{self.sgzt[fund_id]}\n")
        if os.path.exists(self.sgzt_file):
            os.remove(self.sgzt_file)
        os.rename(self.sgzt_file_tmp, self.sgzt_file)

    def load_update_log(self):
        if not os.path.exists(self.update_log_file):
            if os.path.exists(self.update_log_file_tmp):
                os.rename(self.update_log_file_tmp, self.update_log_file)
            else:
                f = open(self.update_log_file, 'w')
                f.close()
        with open(self.update_log_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                fund_id, update_date, op = line.strip().split(",")
                if op == "beg":
                    self.update_log[fund_id] = None
                elif op == "end":
                    self.update_log[fund_id] = update_date
                else:
                    raise ValueError(f"op {op} not supported yet")

    def save_update_log(self):
        with open(self.update_log_file_tmp, "w") as f:
            for fund_id in self.update_log:
                if self.update_log[fund_id] is not None:
                    f.write(f"{fund_id},{self.update_log[fund_id]},end\n")
        if os.path.exists(self.update_log_file):
            os.remove(self.update_log_file)
        os.rename(self.update_log_file_tmp, self.update_log_file)

    def get_fund_ids(self):
        url_api = "http://fund.eastmoney.com/Data/Fund_JJJZ_Data.aspx?t=1&lx=1&letter=&gsid=&text=&sort=zdf,desc&page=1,2000000000&dt={}&atfc=&onlySale=0"
        fund_ids_url = "http://fund.eastmoney.com/fund.html"
        self.driver.get(fund_ids_url)
        dt = self.driver.execute_script("return new Date().getTime();")
        url_api = url_api.format(dt)
        self.driver.get(url_api)
        js = self.driver.find_element_by_tag_name("body").text
        js = js + ";return db;"
        result = self.driver.execute_script(js)
        fund_ids = [d[0] for d in result['datas']]
        return fund_ids

    def load_lsjz(self):
        today = str(datetime.date.today())
        fund_ids = self.fund_ids
        update_date = []
        for fund_id in fund_ids:
            if fund_id not in self.update_log or self.update_log[fund_id] is None:
                update_date.append('')
            else:
                last_timestamp = datetime.datetime.strptime(self.update_log[fund_id], "%Y-%m-%d")
                next_timestamp = last_timestamp + datetime.timedelta(days=1)
                update_date.append(next_timestamp.strftime("%Y-%m-%d"))
        driver = self.driver
        url = "http://fundf10.eastmoney.com/jjjz_290012.html"
        driver.get(url)
        with open("load_jz.js", "r") as f:
            js_script = f.read()
        driver.execute_script(js_script)

        def lsjz_loader():
            driver.execute_script(f"spider_fund_ids={fund_ids};")
            driver.execute_script(f"spider_update_date={update_date};")
            driver.execute_script("load_jjjz_multi_workers()")
            finished = False
            while not finished:
                res = driver.execute_script("return get_jjjz();")
                yield res
                finished = driver.execute_script("return spider_running_workers==0&&spider_buf.length==0;")

        with open(self.sgzt_file, 'a') as sgzt_file:
            with open(self.update_log_file, 'a') as update_log_file:
                for res in lsjz_loader():
                    if len(res) > 0:
                        for fund_id, data in res:
                            data = data[::-1]
                            for i, item in enumerate(data):
                                if item['LJJZ'] == "":
                                    item['LJJZ'] = data[i - 1]['LJJZ']
                                if item['DWJZ'] == "":
                                    item['DWJZ'] = data[i - 1]['DWJZ']
                            if len(data)>0:
                                if data[-1]['SGZT']=='开放申购' or data[-1]['SGZT']=='限制大额申购':
                                    self.sgzt[fund_id]='True'
                                else:
                                    self.sgzt[fund_id]='False'
                                sgzt_file.write(f"{fund_id},{self.sgzt[fund_id]}\n")
                                sgzt_file.flush()

                            data = ['{},{},{}\n'.format(item['FSRQ'], item['DWJZ'], item['LJJZ']) for item in data]
                            data_file = os.path.join(self.dataset_folder, f"{fund_id}.csv")
                            if os.path.exists(data_file) and (
                                    fund_id not in self.update_log or self.update_log[fund_id] is None):
                                os.remove(data_file)
                            with open(data_file, "a") as f:
                                self.update_log[fund_id] = today
                                update_log_file.write(f"{fund_id},{today},beg\n")
                                update_log_file.flush()
                                f.writelines(data)
                                update_log_file.write(f"{fund_id},{today},end\n")
                                update_log_file.flush()
                    else:
                        time.sleep(0.2)
        self.save_update_log()
        self.save_sgzt()

    def close(self):
        self.driver.quit()
        print("driver closed")


if __name__ == "__main__":
    chrome_driver = "D:/D/chromedriver/chromedriver.exe"
    spider = TianTianFundSpider(chrome_driver, headless=False)  # set headless=True in production enviroment
    spider.load_lsjz()
    spider.close()
