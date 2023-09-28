from django.shortcuts import render
from dotenv import load_dotenv, find_dotenv
import json
import os
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import numpy as np
from django.core.management.base import BaseCommand
from movie.models import Movie

# Create your views here.
def view(request):
    def handle(request,prompt):
        #Se lee del archivo .env la api key de openai
        _ = load_dotenv('../openAI.env')
        openai.api_key  = os.environ['openAI_api_key']
        
        items = Movie.objects.all()

        req = prompt
        emb_req = get_embedding(req,engine='text-embedding-ada-002')

        sim = []
        for i in range(len(items)):
            emb = items[i].emb
            emb = list(np.frombuffer(emb))
            sim.append(cosine_similarity(emb,emb_req))
        sim = np.array(sim)
        idx = np.argmax(sim)
        idx = int(idx)
        return items[idx]
    searchTerm = request.GET.get('searchMovie')
    if searchTerm: 
        term=handle(request,searchTerm)
        movies = Movie.objects.filter(title__icontains=term)
    else:
        return render(request, 'recommendations/home.html')
    return render(request, 'recommendations/home.html', {'searchTerm':searchTerm, 'movies': movies})
