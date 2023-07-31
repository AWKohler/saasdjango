from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import AuthenticationFailed
from rest_framework_api_key.models import APIKey


# class APIKeyAuthentication(BaseAuthentication):
#     def authenticate(self, request):
#         api_key_header = request.META.get('HTTP_AUTHORIZATION')
#         if not api_key_header or 'Api-Key' not in api_key_header:
#             return None
#         _, api_key_string = api_key_header.split()
#         api_key = APIKey.objects.filter(key=api_key_string).first()
#         if not api_key:
#             raise AuthenticationFailed('Invalid API Key.')
#
#         return (api_key.created_by, api_key)

# class APIKeyAuthentication(BaseAuthentication):
#     def authenticate(self, request):
#         api_key_header = request.META.get('HTTP_AUTHORIZATION')
#         if not api_key_header or 'Api-Key' not in api_key_header:
#             return None
#         _, api_key_string = api_key_header.split()
#         api_key = APIKey.objects.filter(hashed_key=api_key_string).first()
#         if not api_key:
#             raise AuthenticationFailed('Invalid API Key.')
#
#         return (api_key.created_by, api_key)

class APIKeyAuthentication(BaseAuthentication):
    def authenticate(self, request):
        api_key_header = request.META.get('HTTP_AUTHORIZATION')
        if not api_key_header or 'Api-Key' not in api_key_header:
            return None

        _, api_key_string = api_key_header.split()
        api_key = APIKey.objects.filter(key=api_key_string).first()

        if not api_key:
            raise AuthenticationFailed('Invalid API Key.')

        return api_key.user, api_key

