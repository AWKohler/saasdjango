from django.http import Http404
# from rest_framework_api_key.models import APIKey
# from billing.models import APIKeyUsage


class APIKeyMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        # avoid import issues by importing here
        from billing.models import APIKeyUsage
        self.APIKeyUsage = APIKeyUsage

    def __call__(self, request):
        api_key_header = request.headers.get('Authorization')
        if api_key_header:
            # avoid import issues by importing here
            from rest_framework_api_key.models import APIKey

            scheme, _, api_key = api_key_header.partition(" ")
            if scheme.lower() == "api-key":
                prefix, _, _= api_key.partition(".")
                if prefix:
                    try:
                        api_key_obj = APIKey.objects.get(prefix=prefix)
                        usage_obj, _ = self.APIKeyUsage.objects.get_or_create(api_key_name=api_key_obj.name)
                        usage_obj.increment()
                    except APIKey.DoesNotExist:
                        raise Http404("Invalid API Key")

        response = self.get_response(request)
        return response

# class APIKeyMiddleware:
#     def __init__(self, get_response):
#         self.get_response = get_response
#
#     def __call__(self, request):
#         api_key_header = request.headers.get('Authorization')
#         if api_key_header:
#             scheme, _, api_key = api_key_header.partition(" ")
#             if scheme.lower() == "api-key":
#                 prefix, _, _ = api_key.partition(".")
#                 if prefix:
#                     try:
#                         api_key_obj = APIKey.objects.get(prefix=prefix)
#                         usage_obj, _ = APIKeyUsage.objects.get_or_create(api_key_name=api_key_obj.name)
#                         usage_obj.increment()
#                     except APIKey.DoesNotExist:
#                         raise Http404("Invalid API Key")
#
#         response = self.get_response(request)
#         return response

# from rest_framework_api_key.models import APIKey
# from billing.models import APIKeyUsage
#
#
# class APIKeyMiddleware:
#     def __init__(self, get_response):
#         self.get_response = get_response
#
#     def __call__(self, request):
#         api_key_header = request.headers.get('HTTP_AUTHORIZATION')
#         if api_key_header:
#
#             # print("api key was used")
#
#             api_key_prefix, _, api_key = api_key_header.partition(".")
#             if api_key_prefix:
#
#                 # print(api_key_prefix + " was used")
#
#                 try:
#                     api_key_obj = APIKey.objects.get(id=api_key_prefix)
#                     if api_key_obj:
#                         # Find the usage instance related to the key
#                         usage_obj = APIKeyUsage.objects.get(api_key=api_key_obj.name)
#                         # Increment the count
#                         usage_obj.count += 1
#                         usage_obj.save()
#                 except APIKey.DoesNotExist:
#                     pass
#         response = self.get_response(request)
#         return response
