import ctypes as ct
import numpy as np
import os


class BlasConfig:
    RowMajor = 101
    ColMajor = 102
    NoTrans = 111
    Trans = 112
    Upper = 121
    Lower = 122
    NonUnit = 131
    Unit = 132


dll_path = r".\libopenblas.dll"

if not os.path.exists(dll_path):
    print(f"Ошибка: Файл библиотеки не найден по пути:\n{dll_path}")
    print("Проверьте правильность пути или наличие файла.")
    exit(1)

try:
    blas_lib = ct.CDLL(dll_path)
    print(f"[OK] Библиотека загружена: {dll_path}")
except OSError as e:
    print(f"Ошибка загрузки библиотеки: {e}")
    if "193" in str(e):
        print("\n!!! ВНИМАНИЕ: Ошибка 193 (%1 is not a valid Win32 application).")
        print("Вы пытаетесь запустить библиотеку архитектуры ARM (woa64) на процессоре Intel/AMD.")
        print("Решение: Скачайте версию 'OpenBLAS-x64.zip' с GitHub Releases и укажите путь к ней.")
    exit(1)


float_ptr = ct.POINTER(ct.c_float)
double_ptr = ct.POINTER(ct.c_double)

signatures = {
    'cblas_sgemv': [ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_float, float_ptr, ct.c_int, float_ptr, ct.c_int, ct.c_float, float_ptr, ct.c_int],
    'cblas_dgemv': [ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_double, double_ptr, ct.c_int, double_ptr, ct.c_int, ct.c_double, double_ptr, ct.c_int],
    'cblas_strmv': [ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, float_ptr, ct.c_int, float_ptr, ct.c_int],
    'cblas_ssymv': [ct.c_int, ct.c_int, ct.c_int, ct.c_float, float_ptr, ct.c_int, float_ptr, ct.c_int, ct.c_float, float_ptr, ct.c_int],
    'cblas_sger':  [ct.c_int, ct.c_int, ct.c_int, ct.c_float, float_ptr, ct.c_int, float_ptr, ct.c_int, float_ptr, ct.c_int],
    'cblas_strsv': [ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, float_ptr, ct.c_int, float_ptr, ct.c_int],
}

for func_name, args_types in signatures.items():
    getattr(blas_lib, func_name).argtypes = args_types

def configure_threads(count):
    """Устанавливает количество потоков для OpenBLAS."""
    blas_lib.openblas_set_num_threads(ct.c_int(count))

def run_validation():
    """Основная функция запуска всех проверок."""
    results_log = []
    test_cases = [
        ("GEMV (float)", validate_gemv_single),
        ("GEMV (double)", validate_gemv_double),
        ("TRMV (Upper)", validate_trmv_op),
        ("SYMV (Symmetric)", validate_symv_op),
        ("GER (Rank-1)", validate_ger_op),
        ("TRSV (Solve)", validate_trsv_op),
    ]

    thread_counts = [1, 4, 8]
    
    print(f"\n{'Тест':<20} | {'Потоки':<6} | {'Статус'}")
    print("-" * 40)

    for name, test_func in test_cases:
        for t_count in thread_counts:
            configure_threads(t_count)
            success = test_func()
            status = "OK" if success else "FAIL"
            results_log.append((name, t_count, status))
            print(f"{name:<20} | {t_count:<6} | {status}")

    return all(r[2] == "OK" for r in results_log)



def validate_gemv_single():
    rows, cols = 3, 2
    matrix = np.asarray([[1., 2.], [3., 4.], [5., 6.]], dtype=np.float32)
    vec_in = np.asarray([1., 2.], dtype=np.float32)
    vec_out = np.zeros(rows, dtype=np.float32)
    
    alpha, beta = 2.0, 1.0
    
    blas_lib.cblas_sgemv(
        BlasConfig.RowMajor, BlasConfig.NoTrans, rows, cols,
        alpha,
        matrix.ctypes.data_as(float_ptr), cols,
        vec_in.ctypes.data_as(float_ptr), 1,
        beta,
        vec_out.ctypes.data_as(float_ptr), 1
    )
    
    target = alpha * (matrix @ vec_in) + beta * np.zeros_like(vec_out)
    return np.allclose(vec_out, target)

def validate_gemv_double():
    rows, cols = 3, 2
    matrix = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    vec_in = np.array([1, 2], dtype=np.float64)
    vec_out = np.zeros(rows, dtype=np.float64)
    
    scale = 2.0
    
    blas_lib.cblas_dgemv(
        BlasConfig.RowMajor, BlasConfig.NoTrans, rows, cols,
        scale,
        matrix.ctypes.data_as(double_ptr), cols,
        vec_in.ctypes.data_as(double_ptr), 1,
        1.0,
        vec_out.ctypes.data_as(double_ptr), 1
    )
    
    expected = scale * (matrix @ vec_in)
    return np.allclose(vec_out, expected)

def validate_trmv_op():
    dim = 3
    
    matrix = np.triu(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32))
    vec = np.array([1., 2., 3.], dtype=np.float32)
    vec_backup = vec.copy()
    
    blas_lib.cblas_strmv(
        BlasConfig.RowMajor, BlasConfig.Upper, BlasConfig.NoTrans, BlasConfig.NonUnit,
        dim,
        matrix.ctypes.data_as(float_ptr), dim,
        vec.ctypes.data_as(float_ptr), 1
    )
    
    expected = matrix @ vec_backup
    return np.allclose(vec, expected)

def validate_symv_op():
    dim = 3
    
    base = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]], dtype=np.float32)
    vec = np.array([1., 2., 3.], dtype=np.float32)
    res = np.zeros(dim, dtype=np.float32)
    
    alpha = 2.0
    
    blas_lib.cblas_ssymv(
        BlasConfig.RowMajor, BlasConfig.Upper, dim,
        alpha,
        base.ctypes.data_as(float_ptr), dim,
        vec.ctypes.data_as(float_ptr), 1,
        1.0,
        res.ctypes.data_as(float_ptr), 1
    )
    
    expected = alpha * (base @ vec)
    return np.allclose(res, expected)

def validate_ger_op():
    m, n = 3, 2
    matrix = np.full((m, n), 2.0, dtype=np.float32)
    vec_x = np.array([1., 2., 3.], dtype=np.float32)
    vec_y = np.array([4., 5.], dtype=np.float32)
    
    alpha = 2.0
    
    blas_lib.cblas_sger(
        BlasConfig.RowMajor, m, n,
        alpha,
        vec_x.ctypes.data_as(float_ptr), 1,
        vec_y.ctypes.data_as(float_ptr), 1,
        matrix.ctypes.data_as(float_ptr), n
    )
    
    expected = np.full((m, n), 2.0) + alpha * np.outer(vec_x, vec_y)
    return np.allclose(matrix, expected)

def validate_trsv_op():
    dim = 3
    
    matrix = np.array([[2, 1, 1], [0, 2, 1], [0, 0, 2]], dtype=np.float32)
    rhs = np.array([4., 5., 6.], dtype=np.float32)
    rhs_orig = rhs.copy()
    
    blas_lib.cblas_strsv(
        BlasConfig.RowMajor, BlasConfig.Upper, BlasConfig.NoTrans, BlasConfig.NonUnit,
        dim,
        matrix.ctypes.data_as(float_ptr), dim,
        rhs.ctypes.data_as(float_ptr), 1
    )
    
    expected = np.linalg.solve(matrix, rhs_orig)
    return np.allclose(rhs, expected)

if __name__ == "__main__":
    print("=" * 40)
    print("Запуск набора тестов OpenBLAS (ctypes)")
    print("=" * 40)
    
    if run_validation():
        print("\n[SUCCESS] Все проверки пройдены успешно.")
    else:
        print("\n[ERROR] Обнаружены несоответствия в вычислениях.")
    
    print("=" * 40)