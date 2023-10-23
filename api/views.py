import os

import openai as openai
from PyPDF2 import PdfReader
from django.contrib.auth.models import AnonymousUser
from django.http import StreamingHttpResponse, JsonResponse, HttpResponse
from django.utils.decorators import method_decorator
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from rest_framework.decorators import permission_classes, authentication_classes
from gradio_client import Client
from rest_framework_api_key.permissions import HasAPIKey
from rest_framework.decorators import parser_classes
from rest_framework.parsers import FileUploadParser
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from rest_framework import viewsets

from .APIKeyAuthentication import APIKeyAuthentication
from .CustomJWTAuthentication import CustomJWTAuthentication
from .serializers import BotSerializer
from .models import Bot
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.permissions import IsAuthenticated
import os
import PyPDF2
from docx import Document
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from rest_framework.decorators import api_view, permission_classes, authentication_classes
from rest_framework.permissions import IsAuthenticated

from django.shortcuts import render, get_object_or_404
from django.views.generic import View
from .models import Bot
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.contrib import messages


@api_view(['GET'])
def apiOverview(request):
    api_urls = {
        'List': '/task-list/',
    }

    return Response(api_urls)


@api_view(['GET'])
# @permission_classes([HasAPIKey])
# @authentication_classes([HasAPIKey])
@permission_classes([IsAuthenticated])
@authentication_classes([JWTAuthentication])
def get_user(request):
    return Response(request.user.__str__())


@api_view(["POST"])
# @permission_classes([HasAPIKey])
@permission_classes([HasAPIKey, JWTAuthentication])
def generate_names(request):
    if request.method == 'POST':
        # Parse the request body and extract the prompt
        prompt = request.data.get('prompt')

        # Set up the OpenAI API client
        openai.api_key = "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"

        # Define a generator function to stream the response
        def generate_response():
            for chunk in openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }],
                    stream=True,
            ):
                content = chunk["choices"][0].get("delta", {}).get("content")
                if content is not None:
                    yield content

        # Return a streaming response to the client
        return StreamingHttpResponse(generate_response(), content_type='text/event-stream')

    # Return a JSON error if the request method is not POST
    return JsonResponse({'error': 'Method not allowed.'}, status=405)


@api_view(["POST"])
def falcon(request):
    if request.method == 'POST':
        # Parse the request body and extract the prompt
        prompt = request.data.get('prompt')

        client = Client("https://tiiuae-falcon-chat.hf.space/")
        result = client.predict(
            # "Howdy!",  # str  in 'Click on any example and press Enter in the input textbox!' Dataset component
            prompt,  # str  in 'Click on any example and press Enter in the input textbox!' Dataset component
            fn_index=0
        )
        return Response(result)

    # Return a JSON error if the request method is not POST
    return JsonResponse({'error': 'Method not allowed.'}, status=405)


@api_view(['POST'])
def init_bot(request):
    if request.method == 'POST':
        bot_name = request.POST.get('bot_name')

        # Assuming the bot folder structure is "media/<bot_name>/documents/"
        base_path = os.path.join('media', bot_name)
        documents_path = os.path.join(base_path, 'documents')
        db_path = os.path.join(base_path, 'db')
        merged_file_path = os.path.join(db_path, 'merged.txt')

        csv_files = [file for file in os.listdir(documents_path) if file.endswith('.csv')]
        json_files = [file for file in os.listdir(documents_path) if file.endswith('.json')]
        txt_files = [file for file in os.listdir(documents_path) if file.endswith('.txt')]
        pdf_files = [file for file in os.listdir(documents_path) if file.endswith('.pdf')]
        docx_files = [file for file in os.listdir(documents_path) if file.endswith('.docx')]

        merged_content = ""

        for file_path in csv_files:
            file_path = os.path.join(documents_path, file_path)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                file_content = file.read()
                merged_content += file_content

        for file_path in json_files:
            file_path = os.path.join(documents_path, file_path)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                file_content = file.read()
                merged_content += file_content

        for file_path in txt_files:
            file_path = os.path.join(documents_path, file_path)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                file_content = file.read()
                merged_content += file_content

        for file_path in pdf_files:
            file_path = os.path.join(documents_path, file_path)
            with open(file_path, 'rb') as file:
                pdf = PdfReader(file)
                for page_num in range(len(pdf.pages)):
                    page = pdf.pages[page_num]
                    page_text = page.extract_text()
                    merged_content += page_text

        for file_path in docx_files:
            file_path = os.path.join(documents_path, file_path)
            doc = Document(file_path)
            for paragraph in doc.paragraphs:
                paragraph_text = paragraph.text
                merged_content += paragraph_text

        with open(merged_file_path, 'w') as merged_file:
            merged_file.write(merged_content)

        # Vectorization
        os.environ["OPENAI_API_KEY"] = "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"

        # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"]
        )

        texts = text_splitter.split_text(merged_content)

        embeddings = OpenAIEmbeddings()
        metadatas = [{"source": f"Text chunk {i} of {len(texts)}"} for i in range(len(texts))]
        docsearch = Chroma.from_texts(texts, embeddings, metadatas=metadatas,
                                      persist_directory=db_path)  # Changes persist directory
        docsearch.persist()

        return HttpResponse(f"Bot '{bot_name}' initiated successfully.")


# @api_view(['POST'])
# @permission_classes([IsAuthenticated])
# @authentication_classes([JWTAuthentication])

@api_view(['POST'])
def query_bot(request):
    if request.method == "POST":
        bot_name = request.POST.get('bot_name')
        user_input = request.POST.get('query')

        # Set the OPENAI_API_KEY
        os.environ["OPENAI_API_KEY"] = "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"

        # Relative path of the bot's document directory
        base_path = os.path.join('media', bot_name)
        db_path = os.path.join(base_path, 'db')

        # Create embeddings
        embeddings = OpenAIEmbeddings()

        # Load Chroma instance from existing database
        docsearch = Chroma(persist_directory=db_path, embedding_function=embeddings)

        # Create chain instance with OpenAI and chosen chain_type
        chain = RetrievalQAWithSourcesChain.from_chain_type(OpenAI(temperature=0), chain_type="stuff",
                                                            retriever=docsearch.as_retriever())

        # Query the chain
        result = chain({"question": user_input}, return_only_outputs=True)

        # Extract the answer and source information
        answer = result["answer"].replace('\n', ' ')
        source = result["sources"]

        # Return a response in JSON format
        return JsonResponse({
            'bot_name': bot_name,
            'query': user_input,
            'answer': answer,
            'source': source
        })

    return JsonResponse({'error': 'Invalid Method'})
    # return HttpResponse({'error': 'Invalid Method'})


@api_view(['POST'])
@permission_classes([IsAuthenticated])
@authentication_classes([JWTAuthentication])
def query_bot_s(request):
    if request.method == "POST":
        bot_name = request.POST.get('bot_name')
        user_input = request.POST.get('query')

        # Set the OPENAI_API_KEY
        os.environ["OPENAI_API_KEY"] = "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"

        # Relative path of the bot's document directory
        base_path = os.path.join('media', bot_name)
        db_path = os.path.join(base_path, 'db')

        # Create embeddings
        embeddings = OpenAIEmbeddings()

        # Load Chroma instance from existing database
        docsearch = Chroma(persist_directory=db_path, embedding_function=embeddings)

        # Create chain instance with OpenAI and chosen chain_type
        chain = RetrievalQAWithSourcesChain.from_chain_type(OpenAI(temperature=0), chain_type="stuff",
                                                            retriever=docsearch.as_retriever())

        # Query the chain
        result = chain({"question": user_input}, return_only_outputs=True)

        # Extract the answer and source information
        answer = result["answer"].replace('\n', ' ')
        source = result["sources"]

        # Set up the OpenAI API client
        openai.api_key = "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"

        prompt = "Context: {} Question: {}"
        # prompt = '''Context: {} Question: {}'''

        sex = prompt.format(answer, user_input)

        # Define a generator function to stream the response
        def generate_response_now():
            for chunk in openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{
                        "role": "user",
                        "content": sex
                    }],
                    stream=True,
            ):
                content = chunk["choices"][0].get("delta", {}).get("content")
                if content is not None:
                    yield content

        # Return a streaming response to the client
        return StreamingHttpResponse(generate_response_now(), content_type='text/event-stream')

        # Return a JSON error if the request method is not POST
    return JsonResponse({'error': 'Method not allowed.'}, status=405)


@method_decorator(csrf_exempt, name='dispatch')
class BotGetView(View):
    def get(self, request):
        bot = Bot.objects.filter('id')[0]
        # Preparing data for JSON response
        bot_list = list(bot.values("id", "owner", "name", "use", "initialised"))  # convert QuerySet to list of dicts
        return JsonResponse(bot_list, safe=False)  # Return HTTP JSON response


# @method_decorator(csrf_exempt, name='dispatch')
# class BotDetailView(View):
#     def get(self, request, bot_id):
#         bot = get_object_or_404(Bot, pk=bot_id)
#         return JsonResponse({'owner': bot.owner, 'name': bot.name, 'use': bot.use, 'initialised': bot.initialised})

# @method_decorator(csrf_exempt, name='dispatch')  # this is for testing only
# class BotListView(View):
#     def get(self, request):
#         owner = request.GET.get('owner')
#         bots = Bot.objects.filter(owner=owner)
#         # Render these bots in a template or send JSON response based on your need

@method_decorator(csrf_exempt, name='dispatch')
class BotListView(View):
    def get(self, request):
        owner = request.GET.get('owner')
        bots = Bot.objects.filter(owner=owner)
        # Preparing data for JSON response
        bot_list = list(bots.values("id", "owner", "name", "use", "initialised"))  # convert QuerySet to list of dicts
        return JsonResponse(bot_list, safe=False)  # Return HTTP JSON response

@method_decorator(csrf_exempt, name='dispatch')  # this is for testing only
class BotCreateView(View):
    def post(self, request):
        owner = request.POST.get('owner')
        name = request.POST.get('name')
        use = request.POST.get('use')
        init = request.POST.get('init')
        bot = Bot(owner=owner, name=name, use=use, initalised=init)
        bot.save()
        return HttpResponseRedirect(reverse('bot_list') + f'?owner={owner}')


@method_decorator(csrf_exempt, name='dispatch')  # this is for testing only
class BotUpdateView(View):
    def post(self, request, bot_id):
        bot = get_object_or_404(Bot, pk=bot_id, owner=request.POST.get('owner'))
        bot.name = request.POST.get('name')
        bot.use = request.POST.get('use')
        bot.save()
        return HttpResponseRedirect(reverse('bot_list') + f'?owner={bot.owner}')


@method_decorator(csrf_exempt, name='dispatch')  # this is for testing only
class BotDeleteView(View):
    def post(self, request, bot_id):
        bot = get_object_or_404(Bot, pk=bot_id, owner=request.POST.get('owner'))
        bot.delete()
        return HttpResponseRedirect(reverse('bot_list') + f'?owner={bot.owner}')


@method_decorator(csrf_exempt, name='dispatch')  # this is for testing only
class BotIDGetView(View):
    def post(self, request, bot_id):
        # bot = get_object_or_404(Bot, pk=bot_id, owner=request.POST.get('owner'))
        bot = get_object_or_404(Bot, pk=bot_id)

        bot_dict = {
            'id': bot.id,
            'owner': bot.owner,
            'name': bot.name,
            'use': bot.use,
            'initialised': bot.initialised
        }

        return JsonResponse(bot_dict, safe=False)


        # return HttpResponseRedirect(reverse('bot_list') + f'?owner={bot.owner}')
        # return HttpResponseRedirect(reverse('bot_list') + f'?owner={bot.owner}')

# @method_decorator(csrf_exempt, name='dispatch')
# class BotCreateView(View):
#     class BotCreateView(View):
#         def post(self, request):
#             user = request.user if request.user.is_authenticated else None
#             name = request.POST['name']
#             use = request.POST['use']
#             bot = Bot(user=user, name=name, use=use)
#             bot.save()
#             messages.success(request, 'Bot created successfully.')
#             return HttpResponseRedirect(
#                 reverse('bot_list'))  # assuming 'bot_list' is the name of URL where you list bots
#
# # @method_decorator(csrf_exempt, name='dispatch')
# # class BotCreateView(View):
# #     def post(self, request):
# #         bot = Bot(user=request.user, name=request.POST['name'], use=request.POST['use'])
# #         bot.save()
# #         messages.success(request, 'Bot created successfully.')
# #         return HttpResponseRedirect(reverse('bot_list'))  # assuming 'bot_list' is the name of URL where you list bots
#
#
# @method_decorator(csrf_exempt, name='dispatch')
# class BotUpdateView(View):
#     def post(self, request, bot_id):
#         bot = get_object_or_404(Bot, pk=bot_id)
#         bot.name = request.POST['name']
#         bot.use = request.POST['use']
#         bot.save()
#         messages.success(request, 'Bot updated successfully.')
#         return HttpResponseRedirect(reverse('bot_list'))  # assuming 'bot_list' is the name of URL where you list bots
#
#
# @method_decorator(csrf_exempt, name='dispatch')
# class BotDeleteView(View):
#     def post(self, request, bot_id):
#         bot = get_object_or_404(Bot, pk=bot_id)
#         bot.delete()
#         messages.success(request, 'Bot deleted successfully.')
#         return HttpResponseRedirect(reverse('bot_list'))  # assuming 'bot_list' is the name of URL where you list bots
