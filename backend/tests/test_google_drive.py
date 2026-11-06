import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import fitz
from langchain_core.embeddings import Embeddings
from openpyxl import Workbook

from src.config import Settings
from src.index_documents import main
from src.services.google_drive import DriveService, DOC_MIME, FOLDER_MIME, PDF_MIME, XLSX_MIME
from src.services.vectorDB import VectorDB


class TestEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


class GoogleDriveTest(unittest.TestCase):
    def drive(self, folder):
        drive = DriveService.__new__(DriveService)
        drive.folder_id = 'root-folder'
        drive.download_path = Path(folder)
        drive.service = Mock()
        drive.service.files.return_value.get.return_value.execute.return_value = {'mimeType': FOLDER_MIME}
        return drive

    def test_pagination_subfolders_and_unsupported_files(self):
        drive = self.drive('.')
        pdf = {'id': 'pdf1', 'name': 'Manual.pdf', 'mimeType': PDF_MIME}
        doc = {'id': 'doc1', 'name': 'Normas', 'mimeType': DOC_MIME}
        excel = {'id': 'excel1', 'name': 'Perfil.xlsx', 'mimeType': XLSX_MIME}
        api = drive.service.files.return_value
        api.list.return_value.execute.side_effect = [
            {'files': [], 'nextPageToken': 'page2'},
            {'files': [pdf, {'id': 'child', 'mimeType': FOLDER_MIME}]},
            {'files': [doc, excel, {'id': 'image', 'mimeType': 'image/png'}]},
        ]
        self.assertEqual(drive.list_files(), [pdf, doc, excel])
        self.assertEqual(api.list.call_args_list[1].kwargs['pageToken'], 'page2')
        self.assertIn("'child' in parents", api.list.call_args.kwargs['q'])
        self.assertTrue(api.list.call_args.kwargs['includeItemsFromAllDrives'])

    @patch('src.services.google_drive.MediaIoBaseDownload')
    def test_download_export_duplicate_names_and_safe_paths(self, downloader):
        with tempfile.TemporaryDirectory() as folder:
            drive = self.drive(folder)
            drive.list_files = Mock(return_value=[
                {'id': 'pdf1', 'name': '../../Manual.pdf', 'mimeType': PDF_MIME},
                {'id': 'doc1', 'name': '../../Manual.pdf', 'mimeType': DOC_MIME},
                {'id': 'excel1', 'name': 'Perfil.xlsx', 'mimeType': XLSX_MIME},
            ])
            downloader.return_value.next_chunk.side_effect = [(None, False), (None, True), (None, True), (None, True)]
            paths = [Path(p) for p in drive.download_files()]
            self.assertEqual(len(set(paths)), 3)
            self.assertEqual(paths[-1].name, 'Perfil--excel1.xlsx')
            self.assertTrue(all(p.parent == Path(folder) and p.is_file() for p in paths))
            self.assertEqual([call.kwargs['fileId'] for call in drive.service.files.return_value.get_media.call_args_list], ['pdf1', 'excel1'])
            drive.service.files.return_value.export_media.assert_called_once_with(fileId='doc1', mimeType=PDF_MIME)
            self.assertTrue(all(c.args[0].closed for c in downloader.call_args_list))

    @patch('src.services.google_drive.build')
    @patch('src.services.google_drive.service_account.Credentials.from_service_account_info')
    def test_json_credentials_from_dotenv_are_passed_and_hidden(self, credentials, build):
        with tempfile.TemporaryDirectory() as folder:
            env = Path(folder) / '.env'
            env.write_text('GOOGLE_CREDENTIALS_JSON=\'{"private_key":"test-secret"}\'\n', encoding='utf-8')
            settings = Settings(_env_file=env)
            DriveService(str(Path(folder) / 'missing.json'), 'root-folder', folder,
                         settings.GOOGLE_CREDENTIALS_JSON.get_secret_value())
            self.assertEqual(credentials.call_args.args[0], {'private_key': 'test-secret'})
            self.assertNotIn('test-secret', repr(settings))

    def test_invalid_folder_and_empty_folder_fail(self):
        with self.assertRaisesRegex(ValueError, 'apenas o ID'):
            DriveService('missing.json', 'https://drive.google.com/drive/folders/abc', '.')
        drive = self.drive('.')
        drive.service.files.return_value.get.return_value.execute.return_value = {'mimeType': PDF_MIME}
        with self.assertRaisesRegex(ValueError, 'nao aponta para uma pasta'):
            drive.list_files()
        drive.list_files = Mock(return_value=[])
        with self.assertRaisesRegex(ValueError, 'Nenhum PDF, Google Doc ou Excel'):
            drive.download_files()

    @patch('src.services.vectorDB.OpenWebUIEmbeddings', return_value=TestEmbeddings())
    def test_excel_index_keeps_sheet_cells_and_filename(self, embedder):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'Estudante Exémplo.xlsx'
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = 'Perfil'
            sheet.append(['Area de interesse', 'Astronomia'])
            sheet.append(['Cursos concluidos', 0])
            workbook.create_sheet('Vazia')
            workbook.save(path)
            workbook.save(Path(folder) / 'Outro Estudante.xlsx')
            workbook.close()
            database = VectorDB(Settings(_env_file=None, FAISS_INDEX_PATH=str(Path(folder) / 'index')))
            database.create_faiss_index(folder)
            results = database.query('Qual o interesse de Estudante Exemplo?')
            self.assertEqual(len(results), 1)
            doc = results[0]
            self.assertIn('Arquivo: Estudante Exémplo.xlsx', doc.page_content)
            self.assertIn('A1: Area de interesse | B1: Astronomia', doc.page_content)
            self.assertIn('B2: 0', doc.page_content)
            self.assertEqual(doc.metadata['sheet'], 'Perfil')
            self.assertNotIn('page', doc.metadata)

    @patch('src.index_documents.VectorDB')
    @patch('src.index_documents.Settings')
    def test_cli_selects_drive_and_keeps_local_mode(self, settings, database):
        settings.return_value.GOOGLE_DRIVE_FOLDER_ID = 'root-folder'
        with patch('sys.argv', ['index_documents', '--drive']):
            main()
        database.return_value.refresh_knowledge_base.assert_called_once()
        database.return_value.create_faiss_index.assert_not_called()
        with patch('sys.argv', ['index_documents', 'manual.pdf']):
            main()
        database.return_value.create_faiss_index.assert_called_once_with('manual.pdf')
        settings.return_value.GOOGLE_DRIVE_FOLDER_ID = ''
        with patch('sys.argv', ['index_documents', '--drive']), self.assertRaises(SystemExit):
            main()
        with patch('sys.argv', ['index_documents', '--drive', 'manual.pdf']), self.assertRaises(SystemExit):
            main()

    @patch('src.services.vectorDB.OpenWebUIEmbeddings', return_value=TestEmbeddings())
    @patch('src.services.vectorDB.DriveService')
    def test_sync_indexes_downloads_and_preserves_index_on_download_failure(self, service, embedder):
        with tempfile.TemporaryDirectory() as folder:
            settings = Settings(_env_file=None, GOOGLE_DRIVE_FOLDER_ID='root-folder',
                                FAISS_INDEX_PATH=str(Path(folder) / 'index'))
            database = VectorDB(settings)

            def download():
                path = Path(service.call_args.kwargs['download_path']) / 'Manual--id.pdf'
                with fitz.open() as pdf:
                    pdf.new_page().insert_text((72, 72), 'O laboratorio abre as nove horas.')
                    pdf.save(path)
                return [str(path)]

            service.return_value.download_files.side_effect = download
            database.refresh_knowledge_base()
            self.assertFalse(Path(service.call_args.kwargs['download_path']).exists())
            self.assertIn('nove horas', database.query('Horario?')[0].page_content)
            index_path = Path(settings.FAISS_INDEX_PATH)
            before = {p.name: p.read_bytes() for p in index_path.iterdir()}
            service.return_value.download_files.side_effect = RuntimeError('Download falhou')
            with self.assertRaisesRegex(RuntimeError, 'Download falhou'):
                database.refresh_knowledge_base()
            self.assertEqual(before, {p.name: p.read_bytes() for p in index_path.iterdir()})


if __name__ == '__main__':
    unittest.main()
