from selenium.webdriver import Chrome, ChromeOptions
import time
import os

class TianTianFundSpider:
    '''
    remember to call close() when finished
    '''

    def __init__(self, chrome_driver, headless=True):
        option = ChromeOptions()
        if headless:
            option.add_argument("headless")
        self.driver = Chrome(chrome_driver, chrome_options=option)
        self.fund_ids = None
        self.max_fund_num = 0x7fffffff
        if not os.path.exists("./dataset/"):
            os.mkdir("./dataset/")

    def get_fund_ids(self):
        if self.fund_ids is None:
            url_api = "http://fund.eastmoney.com/Data/Fund_JJJZ_Data.aspx?t=1&lx=1&letter=&gsid=&text=&sort=zdf,desc&page=1,2000000000&dt={}&atfc=&onlySale=0"
            fund_ids_url = "http://fund.eastmoney.com/fund.html"
            self.driver.get(fund_ids_url)
            dt = self.driver.execute_script("return new Date().getTime();")
            url_api = url_api.format(dt)
            self.driver.get(url_api)
            js = self.driver.find_element_by_tag_name("body").text
            js = js + ";return db;"
            result = self.driver.execute_script(js)
            self.fund_ids = [d[0] for d in result['datas']]
        return self.fund_ids[:self.max_fund_num]

    def load_lsjz(self):
        fund_ids = self.get_fund_ids()
        driver = self.driver
        url = "http://fundf10.eastmoney.com/jjjz_290012.html"
        driver.get(url)
        with open("load_jz.js", "r") as f:
            js_script = f.read()
        driver.execute_script(js_script)

        def lsjz_loader():
            driver.execute_script(f"spider_fund_ids={fund_ids};")
            driver.execute_script("load_jjjz_multi_workers()")
            finished = False
            while not finished:
                res = driver.execute_script("return get_jjjz();")
                yield res
                finished = driver.execute_script("return spider_running_workers==0;")

        for res in lsjz_loader():
            if len(res)>0:
                for fund_id,data in res:
                    data = ['{},{},{}\n'.format(item['FSRQ'], item['DWJZ'], item['LJJZ']) for item in data]
                    data = data[::-1]
                    with open(f"./dataset/{fund_id}.csv", "w") as f:
                        f.writelines(data)
            else:
                time.sleep(0.1)

    def close(self):
        self.driver.quit()
        print("driver closed")


if __name__=="__main__":
    chrome_driver = "D:/D/chromedriver/chromedriver.exe"
    spider = TianTianFundSpider(chrome_driver, headless=False)
    spider.load_lsjz()
    spider.close()