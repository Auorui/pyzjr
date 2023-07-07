import os
import shutil
import tempfile
from datetime import timedelta
from minio import Minio
from minio.error import S3Error
from minio.error import InvalidResponseError


def Creatclient(endpoint,access_key,secret_key,mode=1):
    """
    :param endpoint:  S3服务的主机名
    :param access_key: 用户ID
    :param secret_key: 密码
    :param mode: 默认为1，自定义，为0，表示使用游乐场测试与开发
    :return: 返回创建的客户端
    """
    if mode==0:
        client = Minio(
                endpoint="play.min.io",
                access_key="Q3AM3UQ867SPQQA43P2F",
                secret_key="zuf+tfteSlswRu7BJ86wekitnifILbZam1KYY3TG",
            )
    else:
        client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
        )
    return client

class Bucket():
    """操作桶"""
    def Foundbucket(self,client,bucket_name):
        """
        :param client: 客户端
        :param bucket_name: 存储桶名
        :return: 存在就继续进行，没有存在就创建一个桶
        """
        found = client.bucket_exists(bucket_name)
        if not found:
            client.make_bucket(bucket_name)
        else:
            print(f"Bucket {bucket_name} already exists")

    def get_bucket_list(self,client):
        """
        列出所有的存储桶名
        """
        try:
           buckets = client.list_buckets()
           for bucket in buckets:
               print(bucket.name, bucket.creation_date)  # 获取桶的名称和创建时间
        except InvalidResponseError as err:
            print(err)

    def get_remove_bucket(self,client,bucket_name):
        """
        删除存储桶
        """
        try:
            client.remove_bucket(bucket_name)
            print("删除存储桶成功")
        except InvalidResponseError as err:
            print(err)

    def get_bucket_files(self,client,bucket_name):
        """
        用于查看存储桶的对象
        """
        try:
            objects = client.list_objects(bucket_name, prefix=None,
                                               recursive=True)
            for obj in objects:
                print(obj.bucket_name, obj.object_name.encode('utf-8'), obj.last_modified,
                      obj.etag, obj.size, obj.content_type)
        except InvalidResponseError as err:
            print(err)

class Object():
    """操作对象
    example:
    MinIO3=MinIO.Object()
    MinIO3.upload_object(client, "ikun", r'D:\Python_zjr\python_minio\doc\ipcs.docx')
    MinIO3.upload_folder(client, "ikun","D:\Python_zjr\python_minio\doc")
    MinIO3.delete_object(client, "ikun",["main2.py","MinIO.py"] )
    MinIO3.delete_folder(client,"ikun","./doc")
    MinIO3.download_folder(client, "ikun", r"folder")
    MinIO.download_objects(client, "ikun",["main2.py","MinIO.py"] )
    """
    def delete_object(self,client, bucket_name, objects):
        """
        删除对象
        :param client:
        :param bucket_name:
        :param objects: 列表，元素为字符，逗号为间隔，如['myobject-1', 'myobject-2', 'myobject-3']
        :return:
        """
        try:
            for obj in objects:
                client.remove_object(bucket_name, obj)
                print(f"Deleted {obj} under {bucket_name}")
        except InvalidResponseError as err:
            print(err)

    def download_object(self,client, bucket_name, objects, filepath=None):
        """
        下载对象
        :param client:
        :param bucket_name:
        :param objects: 列表，元素为字符，逗号为间隔，如["main2.py","MinIO.py"]
        :param filepath: 默认为None，即是运行该文件的根目录下
        :return:
        """
        try:
            if filepath is None:
                current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件的目录
            else:
                current_dir = filepath
            for obj in objects:
                file_path = os.path.join(current_dir, obj)  # 拼接目录和文件名
                client.fget_object(bucket_name, obj, file_path)
                print(f"Downloaded object {obj} to {file_path}")
        except InvalidResponseError as err:
            print(err)

    def upload_object(self,client, bucket_name, file_path):
        """
        上传对象
        :param client:
        :param bucket_name:
        :param file_path: 建议是绝对路径
        :return:
        """
        try:
            object_name = os.path.basename(file_path)  # 提取文件名作为对象名称
            with open(file_path, "rb") as file_data:
                file_size = os.path.getsize(file_path)
                client.put_object(bucket_name, object_name, file_data, file_size)
                print(f"Uploaded object {object_name} to bucket {bucket_name}")
        except IOError as e:
            print(f"Failed to open file: {file_path} - {e}")
        except S3Error as err:
            print(f"Error occurred: {err}")

    def delete_folder(self,client, bucket_name, folder_path):
        """
        删除的也是压缩包
        :param client:
        :param bucket_name:
        :param folder_path:
        :return:
        """
        try:
            temp_dir = tempfile.mkdtemp()
            folder_name = os.path.basename(folder_path)
            zip_file = os.path.join(temp_dir, f"{folder_name}.zip")

            shutil.make_archive(zip_file[:-4], 'zip', folder_path)

            client.remove_object(bucket_name, f"{folder_name}.zip")
            print(f"Deleted {folder_name}.zip from bucket {bucket_name}")

            shutil.rmtree(temp_dir)
        except InvalidResponseError as err:
            print(err)

    def upload_folder(self,client, bucket_name, folder_path):
        """
        上传的文件是zip压缩包
        :param client:
        :param bucket_name:
        :param folder_path:
        :return:
        """
        try:
            temp_dir = tempfile.mkdtemp()
            folder_name = os.path.basename(folder_path)
            zip_file = os.path.join(temp_dir, f"{folder_name}.zip")

            shutil.make_archive(zip_file[:-4], 'zip', folder_path)

            with open(zip_file, "rb") as file_data:
                file_size = os.path.getsize(zip_file)
                client.put_object(bucket_name, f"{folder_name}.zip", file_data, file_size)
                print(f"Uploaded {folder_name}.zip to bucket {bucket_name}")

            shutil.rmtree(temp_dir)
        except IOError as e:
            print(f"Failed to compress folder: {folder_path} - {e}")
        except S3Error as err:
            print(f"Error occurred: {err}")

    def download_folder(self,client, bucket_name, local_path):
        """
        下载后会自动解压
        :param client:
        :param bucket_name:
        :param local_path:
        :return:
        """
        try:
            temp_dir = tempfile.mkdtemp()

            client.fget_object(bucket_name, f"{local_path}.zip", os.path.join(temp_dir, f"{local_path}.zip"))
            print(f"Downloaded {local_path}.zip from bucket {bucket_name}")

            shutil.unpack_archive(os.path.join(temp_dir, f"{local_path}.zip"), local_path)
            print(f"Extracted {local_path}.zip to {local_path}")

            shutil.rmtree(temp_dir)
        except InvalidResponseError as err:
            print(err)

class Presigned():
    def upload_url(self, client, bucket_name, object_name, expires_in=7200):
        """
        生成预签名的上传 URL
        :param client: MinIO 客户端
        :param bucket_name: 存储桶名称
        :param object_name: 对象名称
        :param expires_in: URL 的有效期（单位：秒），默认为 2 小时
        :return: 预签名的上传 URL
        """
        try:
            # 生成预签名 URL
            url = client.presigned_put_object(bucket_name, object_name, expires=timedelta(seconds=expires_in))
            return url
        except Exception as e:
            print(f"Failed to generate presigned upload URL: {e}")
            return None

    def download_url(self,client, bucket_name, object_name, expires_in=7200):
        """
        生成预签名的下载 URL
        :param client: MinIO 客户端
        :param bucket_name: 存储桶名称
        :param object_name: 对象名称
        :param expires_in: URL 的有效期（单位：秒），默认为 2 小时
        :return: 预签名的下载 URL
        """
        try:
            # 生成预签名 URL
            url = client.presigned_get_object(bucket_name, object_name, expires=timedelta(seconds=expires_in))
            return url
        except Exception as e:
            print(f"Failed to generate presigned download URL: {e}")
            return None