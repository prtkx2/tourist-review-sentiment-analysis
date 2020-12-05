from flask import Flask,render_template,request,jsonify
from bs4 import BeautifulSoup
import requests
import csv

app = Flask(__name__)

global location, attraction, reviewLinkTemp, reviewLink, reviews

@app.route('/', methods= ['GET'])
def index():
    return render_template('searchform.html')


@app.route('/search', methods = ['GET', 'POST'])
def getreviews():
    links = []
    location = request.form['Location']
    attraction = request.form['attraction']
    url = f'https://www.google.com/search?q={attraction}+reviews+site%3Awww.tripadvisor.in&oq=qutub+minar+reviews+site%3Awww.tripadvisor.in'
    source = requests.get(url).text
    soup = BeautifulSoup(source, 'lxml')
    for link in soup.find_all('a', limit=17):

        links.append(link.get('href'))
        
    # print(links[len(links)-1])
    
    reviewLinkTemp = links[len(links)-1].split('/url?q=')
    reviewLink = reviewLinkTemp[1] + "#REVIEWS"
    # print(reviewLink)
    revSource = requests.get(reviewLink).text
    revSoup = BeautifulSoup(revSource, 'lxml')
    reviews = []
    for revs in revSoup.select('div[class*="location-review-"]'):
         print(revs)
        
    return jsonify(reviewLink)
    
if __name__ == '__main__':
    app.run()