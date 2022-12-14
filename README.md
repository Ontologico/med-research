# Лидеры цифровой трансформации 2022. Генерация медицинских исследований на основе размеченных патологий.

##  Содержание:
1.	code.ipynb – jupyter notebook с презентацией возможностей собственного модуля.
2.	mycode.py – модуль с необходимыми функциями.
3.	Видео-презентация.wmv – видео с результатом некоторых функций из модуля. Описывает правильность работы json-файла, сгенерированного по маскам. Там показана загрузка исходных изображений и json-файла, полученного из масок. Для просмотра разметки буду использовать данный инструмент: VGG Image Annotator (находится в папке).
4.	data – папка с данными. По умолчанию содержит 3 файла:  
    -	dcmimage.dcm – шаблон dicom, для создания на его основе новых dicom-файлов  
    -	00000057_brain_flair.nii – тестовый набор данных патологий головного мозга  
    -	00000057_final_seg.nii – маски патологий
#### Все остальные файлы в папке «data» появляются в процессе выполнения кода.
## Код решает следующие задачи: 
1.	Создание исходных изображений и изображений масок в форматах jpg, png.
2.	Конвертация масок в json, представленный в формате COCO или VGG. В нашем случае, для единообразия выбран VGG.
3.	Конвертация json в фотографии масок. Эта функция понадобиться после того, как человек разметит фотографию в веб-инструменте разметки.
4.	Конвертация изображений масок или исходных изображений в формат nifti.
5.	Конвертация nifty в dicom и dicom в nifti.
6.	Конвертация dicom в изображение.
#### Последовательность вызовов функций построена таким образом, чтобы сразу продемонстрировать «конвейерную» работу кода, так как на каждом этапе используются результаты предыдущего.
#### Дополнительная справочная информация представлена в ячейках «code.ipynb», а также в строках документации к функциям.
