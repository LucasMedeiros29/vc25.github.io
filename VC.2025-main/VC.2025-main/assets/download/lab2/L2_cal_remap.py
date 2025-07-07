import cv2
import numpy as np
import os
import glob

# ===========================================================================
# MODIFICAÇÃO 1: Ajuste das dimensões do tabuleiro de xadrez para (8,6)
# Conforme especificado no PDF do laboratório. O original era (6,8).
# ===========================================================================
CHECKERBOARD = (8,6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Vetores para armazenar os pontos 3D do mundo real e os pontos 2D da imagem.
objpoints = []  # Pontos 3D no espaço do mundo real
imgpoints = []  # Pontos 2D no plano da imagem

# Definindo as coordenadas do mundo para os pontos 3D
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# ===========================================================================
# MODIFICAÇÃO 2: Alteração do padrão de busca para as imagens .jpg fornecidas
# ===========================================================================
images = glob.glob('frm_cameradistorcida*.jpg')

print(f"Encontradas {len(images)} imagens para calibração.")

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Erro ao ler a imagem: {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Encontra os cantos do tabuleiro de xadrez
    ret, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        h, w = img.shape[:2]
        scale_factor = 800 / w
        img_display = cv2.resize(img, (int(w * scale_factor), int(h * scale_factor)))
        cv2.imshow('Cantos Encontrados', img_display)
        print(f"Cantos encontrados com sucesso em: {fname}")
        cv2.waitKey(500)
    else:
        print(f"Cantos não encontrados em: {fname}")

cv2.destroyAllWindows()

if len(objpoints) == 0 or len(imgpoints) == 0:
    print("\nCalibração falhou. Não foram encontrados cantos em nenhuma imagem.")
    exit()

print(f"\nIniciando a calibração com {len(objpoints)} imagens válidas...")

h, w = gray.shape[:2]

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

print("\n--- Resultados da Calibração ---")
print("Calibração concluída com sucesso!" if ret else "A calibração falhou.")
print("\nMatriz da Câmera (Intrínseca):\n", mtx)
print("\nCoeficientes de Distorção:\n", dist)

# ===========================================================================
# MODIFICAÇÃO 3: Salvar os parâmetros da câmera para uso futuro
# ===========================================================================
np.savez('calib_params.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
print("\nParâmetros de calibração salvos em 'calib_params.npz'")

# Calculando o erro de reprojeção
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

total_error_rms = np.sqrt(mean_error / len(objpoints))
print("\nErro Total (RMS - Raiz do Erro Quadrático Médio): {}".format(total_error_rms))
print("----------------------------------")

# ===========================================================================
# MODIFICAÇÃO 4 (ALTERADA): Corrigir as imagens com cv2.remap() e salvar
# ===========================================================================
print("\nAplicando correção de distorção (com remapping) e salvando imagens corrigidas...")
output_dir = "imagens_corrigidas"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for fname in images:
    img = cv2.imread(fname)
    h, w = img.shape[:2]

    # Refina a matriz da câmera
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # Gera os mapas de remapeamento (mapx, mapy)
    mapx, mapy = cv2.initUndistortRectifyMap(
        mtx, dist, None, newcameramtx, (w, h), cv2.CV_32FC1)

    # Aplica o remapeamento
    dst = cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_LINEAR)

    # Recorta a imagem (opcional, mas recomendado)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    # Salva a imagem corrigida
    output_filename = os.path.join(output_dir, "corrigida_" + os.path.basename(fname))
    cv2.imwrite(output_filename, dst)
    print(f"Salva: {output_filename}")

print("\nProcesso concluído. As imagens corrigidas estão na pasta 'imagens_corrigidas'.")