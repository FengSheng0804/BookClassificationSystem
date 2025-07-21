# 小说：http://read.nlc.cn/yuewen/index?&pageNo=1&categoryId=13100
# 文学：http://read.nlc.cn/yuewen/index?&pageNo=1&categoryId=14100
# 医学：http://read.nlc.cn/yuewen/index?&pageNo=1&categoryId=15300
# 历史：http://read.nlc.cn/yuewen/index?&pageNo=1&categoryId=14400
# 艺术：http://read.nlc.cn/yuewen/index?&pageNo=1&categoryId=15200
# 养生：http://read.nlc.cn/yuewen/index?&pageNo=1&categoryId=13800
# 心理：http://read.nlc.cn/yuewen/index?&pageNo=1&categoryId=14000
# 科普：http://read.nlc.cn/yuewen/index?&pageNo=1&categoryId=16300
# 军事：http://read.nlc.cn/yuewen/index?&pageNo=1&categoryId=14600
# 旅游：http://read.nlc.cn/yuewen/index?&pageNo=1&categoryId=13900

"""
图书封面爬虫脚本 - 从国家图书馆爬取不同类别的图书封面
"""

import requests
from bs4 import BeautifulSoup
import os
import time
import random
from urllib.parse import urljoin
from PIL import Image
import io
import logging
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 图书分类配置
BOOK_CATEGORIES = {
    'novel': {'name': '小说', 'categoryId': '13100'},
    'literature': {'name': '文学', 'categoryId': '14100'}, 
    'medicine': {'name': '医学', 'categoryId': '15300'},
    'history': {'name': '历史', 'categoryId': '14400'},
    'art': {'name': '艺术', 'categoryId': '15200'},
    'health': {'name': '养生', 'categoryId': '13800'},
    'psychology': {'name': '心理', 'categoryId': '14000'},
    'science': {'name': '科普', 'categoryId': '16300'},
    'military': {'name': '军事', 'categoryId': '14600'},
    'travel': {'name': '旅游', 'categoryId': '13900'}
}

# 用户代理列表（模拟不同浏览器）
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/120.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15'
]

# 爬虫配置
BASE_URL = 'http://read.nlc.cn/yuewen/index'
DATASET_DIR = './multimodal_classificate/dataset'
MAX_PAGES = 100  # 每个类别爬取的最大页数
DELAY_RANGE = (1, 3)  # 请求间延时范围（秒）
MAX_RETRIES = 3  # 最大重试次数
TIMEOUT = 30  # 请求超时时间

class BookCoverCrawler:
    """图书封面爬虫类"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
    
    def get_random_user_agent(self):
        """获取随机User-Agent"""
        return random.choice(USER_AGENTS)
    
    def safe_delay(self):
        """安全延时"""
        delay = random.uniform(*DELAY_RANGE)
        time.sleep(delay)
    
    def create_category_dir(self, category_key):
        """创建分类目录"""
        category_name = BOOK_CATEGORIES[category_key]['name']
        category_dir = os.path.join(DATASET_DIR, category_key)
        os.makedirs(category_dir, exist_ok=True)
        logger.info(f"创建目录: {category_dir} ({category_name})")
        return category_dir
    
    def fetch_page(self, url, retries=0):
        """获取页面内容"""
        if retries >= MAX_RETRIES:
            logger.error(f"达到最大重试次数，跳过URL: {url}")
            return None
        
        try:
            # 设置随机User-Agent
            headers = {'User-Agent': self.get_random_user_agent()}
            
            # 发送请求
            response = self.session.get(url, headers=headers, timeout=TIMEOUT)
            response.raise_for_status()
            
            return response
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"请求失败 (尝试 {retries + 1}/{MAX_RETRIES}): {url}, 错误: {e}")
            time.sleep(2 ** retries)  # 指数退避
            return self.fetch_page(url, retries + 1)
    
    def parse_book_covers(self, html_content):
        """解析页面中的图书封面链接和书名"""
        soup = BeautifulSoup(html_content, 'html.parser')
        book_info_list = []
        
        # 查找所有书籍条目
        book_items = soup.find_all('li')
        
        for item in book_items:
            # 查找书名
            title_span = item.find('span', class_='tt')
            if not title_span:
                continue
                
            title = title_span.get_text(strip=True)
            
            # 查找封面图片
            img_tag = item.find('img')
            if not img_tag:
                continue
                
            src = img_tag.get('src') or img_tag.get('data-src')
            if src and any(ext in src.lower() for ext in ['.webp', '.jpg', '.jpeg', '.png']):
                # 处理相对URL
                if src.startswith('//'):
                    src = 'http:' + src
                elif src.startswith('/'):
                    src = 'http://read.nlc.cn' + src
                
                book_info_list.append({'title': title, 'cover_url': src})
        
        logger.info(f"在页面中找到 {len(book_info_list)} 本书的信息")
        return book_info_list
    
    def save_titles_to_file(self, category_dir, title_info_list):
        """将书名列表保存到txt文件"""
        title_file_path = os.path.join(category_dir, 'titles.txt')
        
        try:
            with open(title_file_path, 'a', encoding='utf-8') as f:
                for title_info in title_info_list:
                    # 删除书名中的所有括号及其内容
                    clean_title = self.clean_title(title_info['title'])
                    f.write(f"{title_info['filename']}:{clean_title}\n")
            logger.debug(f"已保存 {len(title_info_list)} 个书名到: {title_file_path}")
        except Exception as e:
            logger.error(f"保存书名到文件失败: {e}")
    
    def clean_title(self, title):
        """清理书名，删除所有括号及其内容和标点符号"""
        import re
        import string
        
        # 删除所有类型的括号及其内容：()、[]、{}、（）、【】、『』、「」
        bracket_patterns = [
            r'\([^)]*\)',      # 英文圆括号
            r'\（[^）]*\）',    # 中文圆括号
            r'\[[^\]]*\]',     # 方括号
            r'\【[^】]*\】',    # 中文方括号
            r'\{[^}]*\}',      # 花括号
            r'\『[^』]*\』',    # 中文书名号1
            r'\「[^」]*\」'     # 中文书名号2
        ]
        
        clean_title = title
        # 删除括号及其内容
        for pattern in bracket_patterns:
            clean_title = re.sub(pattern, '', clean_title)
        
        # 删除所有标点符号
        # 英文标点符号
        clean_title = clean_title.translate(str.maketrans('', '', string.punctuation))
        
        # 中文标点符号
        chinese_punctuation = '！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞？＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〽゛゜ゝゞ・“”·‐‑‒–—―‖‗''""†‡•‰′″‴※‵‶‷‸‹›‼‽‾‿⁀⁁⁂⁃⁅⁆⁇⁈⁉⁊⁋⁌⁍⁎⁏⁐⁑⁒⁓⁔⁕⁖⁗⁘⁙⁚⁛⁜⁝⁞⁺⁻⁼⁽⁾ⁿ₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ℂℇℊℋℌℍℎℏℐℑℒℓℕ№℗℘ℙℚℛℜℝ℞℟℠℡™ℨ℩KÅℬℭ℮ℯℰℱℲℳℴℵℶℷℸℹ℺℻ℼℽℾℿ⅀⅁⅂⅃⅄ⅅⅆⅇⅈⅉ⅊⅋⅌⅍ⅎ⅏＝≈≠≡≢≤≥≦≧≨≩'
        for punct in chinese_punctuation:
            clean_title = clean_title.replace(punct, '')
        
        # 清理多余的空格并去除首尾空格
        clean_title = re.sub(r'\s+', ' ', clean_title).strip()
        return clean_title
    
    def download_and_convert_image(self, image_url, save_path, retries=0):
        """下载并转换图片为JPG格式"""
        if retries >= MAX_RETRIES:
            logger.error(f"图片下载失败，达到最大重试次数: {image_url}")
            return False
        
        try:
            # 下载图片
            headers = {'User-Agent': self.get_random_user_agent()}
            response = self.session.get(image_url, headers=headers, timeout=TIMEOUT)
            response.raise_for_status()
            
            # 使用PIL处理图片
            image = Image.open(io.BytesIO(response.content))
            
            # 转换为RGB模式（JPG不支持透明通道）
            if image.mode in ('RGBA', 'P'):
                image = image.convert('RGB')
            
            # 保存为JPG
            image.save(save_path, 'JPEG', quality=90)
            logger.debug(f"图片已保存: {save_path}")
            return True
            
        except Exception as e:
            logger.warning(f"图片处理失败 (尝试 {retries + 1}/{MAX_RETRIES}): {image_url}, 错误: {e}")
            time.sleep(1)
            return self.download_and_convert_image(image_url, save_path, retries + 1)
    
    def crawl_category(self, category_key):
        """爬取特定分类的图书封面"""
        category_info = BOOK_CATEGORIES[category_key]
        category_name = category_info['name']
        category_id = category_info['categoryId']
        
        logger.info(f"开始爬取 [{category_name}] 分类，ID: {category_id}")
        
        # 创建分类目录
        category_dir = self.create_category_dir(category_key)
        
        total_downloaded = 0
        image_counter = 1
        pending_title_info = []  # 待保存的书名信息列表（包含文件名和书名）
        
        # 爬取每一页
        for page_no in tqdm(range(1, MAX_PAGES + 1), desc=f"爬取{category_name}"):
            logger.info(f"正在爬取 [{category_name}] 第 {page_no} 页")
            
            # 构建页面URL
            page_url = f"{BASE_URL}?pageNo={page_no}&categoryId={category_id}"
            
            # 获取页面内容
            response = self.fetch_page(page_url)
            if not response:
                logger.warning(f"跳过第 {page_no} 页")
                continue
            
            # 解析封面链接和书名
            book_info_list = self.parse_book_covers(response.text)
            
            if not book_info_list:
                logger.warning(f"第 {page_no} 页未找到图书信息，可能已达到最后一页")
                continue
            
            # 下载每个封面
            for book_info in book_info_list:
                cover_url = book_info['cover_url']
                title = book_info['title']
                
                save_filename = f"{image_counter:06d}.jpg"
                save_path = os.path.join(category_dir, save_filename)
                
                # 跳过已存在的文件
                if os.path.exists(save_path):
                    logger.debug(f"文件已存在，跳过: {save_path}")
                    image_counter += 1
                    continue
                
                # 下载并转换图片
                if self.download_and_convert_image(cover_url, save_path):
                    total_downloaded += 1
                    image_counter += 1
                    # 添加文件名和书名的对应信息到待保存列表
                    pending_title_info.append({
                        'filename': save_filename,
                        'title': title
                    })
                    
                    # 每下载15张图片后保存书名并延时
                    if total_downloaded % 15 == 0:
                        if pending_title_info:
                            self.save_titles_to_file(category_dir, pending_title_info)
                            pending_title_info = []  # 清空待保存列表
                        self.safe_delay()
                else:
                    logger.warning(f"图片下载失败: {cover_url}")
            
            # 页面间延时
            self.safe_delay()
        
        # 保存剩余的书名
        if pending_title_info:
            self.save_titles_to_file(category_dir, pending_title_info)
        
        logger.info(f"[{category_name}] 分类爬取完成，共下载 {total_downloaded} 张图片")
        return total_downloaded
    
    def crawl_all_categories(self):
        """爬取所有分类的图书封面"""
        logger.info("开始爬取所有分类的图书封面")
        logger.info(f"目标：每个分类爬取前 {MAX_PAGES} 页")
        
        # 创建数据集根目录
        os.makedirs(DATASET_DIR, exist_ok=True)
        
        total_images = 0
        results = {}
        
        for category_key in BOOK_CATEGORIES.keys():
            try:
                downloaded = self.crawl_category(category_key)
                results[category_key] = downloaded
                total_images += downloaded
                
                # 分类间增加更长延时
                logger.info(f"分类 [{BOOK_CATEGORIES[category_key]['name']}] 完成，休息片刻...")
                time.sleep(random.uniform(10, 20))
                
            except Exception as e:
                logger.error(f"爬取分类 [{BOOK_CATEGORIES[category_key]['name']}] 时发生错误: {e}")
                results[category_key] = 0
        
        # 输出统计结果
        self.print_summary(results, total_images)
    
    def print_summary(self, results, total_images):
        """打印爬取统计结果"""
        logger.info("="*60)
        logger.info("爬取统计结果")
        logger.info("="*60)
        
        for category_key, count in results.items():
            category_name = BOOK_CATEGORIES[category_key]['name']
            logger.info(f"{category_name}: {count} 张图片")
        
        logger.info("-"*60)
        logger.info(f"总计: {total_images} 张图片")
        logger.info(f"数据集保存路径: {os.path.abspath(DATASET_DIR)}")
        logger.info("="*60)

def check_requirements():
    """检查必要的依赖"""
    try:
        import requests
        import bs4
        import PIL
        import tqdm
        logger.info("✅ 所有依赖检查通过")
        return True
    except ImportError as e:
        logger.error(f"❌ 缺少必要依赖: {e}")
        logger.error("请安装: pip install requests beautifulsoup4 pillow tqdm")
        return False

def main():
    """主函数"""
    print("🕷️  图书封面爬虫")
    print("="*60)
    print("⚠️  注意事项:")
    print("   1. 此脚本仅用于学习研究目的")
    print("   2. 请遵守网站的robots.txt和使用条款")
    print("   3. 爬取过程中请保持网络连接稳定")
    print("   4. 程序已设置延时和重试机制以保护目标网站")
    print("="*60)
    
    # 检查依赖
    if not check_requirements():
        return
    
    # 确认开始爬取
    user_input = input("是否开始爬取? (y/N): ").lower().strip()
    if user_input != 'y':
        print("已取消爬取")
        return
    
    try:
        # 创建爬虫实例
        crawler = BookCoverCrawler()
        
        # 开始爬取
        crawler.crawl_all_categories()
        
        print("\n🎉 爬取任务完成!")
        print(f"📁 数据集保存在: {os.path.abspath(DATASET_DIR)}")
        
    except KeyboardInterrupt:
        print("\n⏹️  用户中断爬取")
    except Exception as e:
        logger.error(f"爬取过程中发生未预期错误: {e}")

if __name__ == "__main__":
    main()