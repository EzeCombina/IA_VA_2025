/*==================[Inclusiones]======================*/
#include <stdio.h>
#include <string.h>
#include <sys/param.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "lwip/sockets.h"
#include "lwip/netdb.h"

#include "../include/motors.h"
#include "../include/cmdParser.h"

/*==================[Definiciones]======================*/
#define PORT 3333
#define PROCESADORA 0
#define PROCESADORB 1
#define VALUE_TEST 65

/*==================[Variables globales]======================*/
static const char *TAG = "TCP_SERVER";

TaskHandle_t wifi_task_handle = NULL;
TaskHandle_t motor_task_handle = NULL;

SemaphoreHandle_t motor_state_mutex;
bool estado_motor = false;

bool estado_motor_copia = false;

// Función para inicializar el modo AP (Access Point)
void wifi_init_softap(void)
{
    esp_netif_create_default_wifi_ap(); // Crea la interfaz de red por defecto para el modo AP
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT(); // Inicializa la configuración de WiFi
    esp_wifi_init(&cfg); // Inicializa el controlador de WiFi

    wifi_config_t wifi_config = {
        .ap = {
            .ssid = "ESP32_AP",                 // Nombre del SSID
            .ssid_len = strlen("ESP32_AP"),     // Longitud del SSID
            .channel = 1,                       // Canal de operación
            .password = "12345678",             // Contraseña del AP
            .max_connection = 1,                // Número máximo de conexiones
            .authmode = WIFI_AUTH_WPA_WPA2_PSK  // Modo de autenticación
        },
    };

    if (strlen("12345678") == 0) wifi_config.ap.authmode = WIFI_AUTH_OPEN;  // Si no hay contraseña, se establece el modo abierto

    esp_wifi_set_mode(WIFI_MODE_AP);                // Establece el modo AP
    esp_wifi_set_config(WIFI_IF_AP, &wifi_config);  // Configura el AP
    esp_wifi_start();                               // Inicia el controlador de WiFi   

    ESP_LOGI(TAG, "Access Point creado. SSID: ESP32_AP, PASS: 12345678");
}

/*==================[Implementacion de la tarea]======================*/

void motor_up_task(void *pvParameters){

    while (1)
    {       
        if (xSemaphoreTake(motor_state_mutex, portMAX_DELAY)) {
            estado_motor_copia = estado_motor;
            xSemaphoreGive(motor_state_mutex);
        }   
             
        if(estado_motor_copia){

        //ESP_LOGI("MOTOR", "Estado actual: %d", estado_motor_copia);
        cmdParser(MOTOR_RIGHT_FORWARD,VALUE_TEST);
        cmdParser(MOTOR_LEFT_FORWARD,VALUE_TEST);
        vTaskDelay(3000/portTICK_PERIOD_MS);

        cmdParser(MOTOR_RIGHT_STOP,VALUE_TEST);
        cmdParser(MOTOR_LEFT_STOP ,VALUE_TEST);
        vTaskDelay(1000/portTICK_PERIOD_MS);

        cmdParser(MOTOR_RIGHT_BACKWARD,VALUE_TEST);
        cmdParser(MOTOR_LEFT_BACKWARD,VALUE_TEST);
        vTaskDelay(3000/portTICK_PERIOD_MS);

        cmdParser(MOTOR_RIGHT_STOP,VALUE_TEST);
        cmdParser(MOTOR_LEFT_STOP ,VALUE_TEST);
        vTaskDelay(1000/portTICK_PERIOD_MS);
        }

        else{
            //ESP_LOGI("MOTOR", "Estado actual: %d", estado_motor_copia);
            cmdParser(MOTOR_RIGHT_STOP,VALUE_TEST);
            cmdParser(MOTOR_LEFT_STOP ,VALUE_TEST);
            vTaskDelay(50 / portTICK_PERIOD_MS);
        }

        vTaskDelay(10 / portTICK_PERIOD_MS);
    }
}

void tcp_server_task(void *pvParameters)
{
    char rx_buffer[128];            // Buffer para recibir datos
    char addr_str[128];             // Buffer para almacenar la dirección IP
    int addr_family = AF_INET;      // IPv4
    int ip_protocol = IPPROTO_IP;   // Protocolo IP

    struct sockaddr_in dest_addr;                   // Estructura para la dirección de destino
    dest_addr.sin_addr.s_addr = htonl(INADDR_ANY);  // Acepta conexiones de cualquier dirección
    dest_addr.sin_family = AF_INET;                 // IPv4
    dest_addr.sin_port = htons(PORT);               // Puerto de escucha

    int listen_sock = socket(addr_family, SOCK_STREAM, ip_protocol);        // Crea el socket
    bind(listen_sock, (struct sockaddr *)&dest_addr, sizeof(dest_addr));    // Asocia el socket a la dirección y puerto
    listen(listen_sock, 1);                                                 // Escucha conexiones entrantes

    ESP_LOGI(TAG, "Esperando conexión TCP en el puerto %d...", PORT);

    while(1){
        struct sockaddr_in6 source_addr; // Estructura para la dirección de origen
        socklen_t addr_len = sizeof(source_addr); // Longitud de la dirección                                //Acá lo modifique
        int sock = accept(listen_sock, (struct sockaddr *)&source_addr, &addr_len); // Acepta la conexión

        inet_ntoa_r(((struct sockaddr_in *)&source_addr)->sin_addr, addr_str, sizeof(addr_str) - 1); // Convierte la dirección IP a una cadena
        ESP_LOGI(TAG, "Cliente conectado desde %s", addr_str); // Imprime la dirección IP del cliente

        while(1)
        {
            int len = recv(sock, rx_buffer, sizeof(rx_buffer) - 1, 0);
            if(len < 0)
            {
                ESP_LOGI(TAG, "Error en recv");
                break;
            }
            
            else if(len == 0)
            {
                ESP_LOGI(TAG, "Cliente desconectado");
                break;
            }
            
            else
            {
                rx_buffer[len] = 0;

                if (strcmp(rx_buffer, "ok") == 0) {
                    if (xSemaphoreTake(motor_state_mutex, portMAX_DELAY)) {
                        estado_motor = true;
                        xSemaphoreGive(motor_state_mutex);
                    }
                    ESP_LOGI(TAG, "Comando recibido: %s", rx_buffer);
                }      

                else if (strcmp(rx_buffer, "stop") == 0) {
                    if (xSemaphoreTake(motor_state_mutex, portMAX_DELAY)) {
                        estado_motor = false;
                        xSemaphoreGive(motor_state_mutex);
                    }
                    ESP_LOGI(TAG, "Comando recibido: %s", rx_buffer);
                }
            }
        }
        close(sock);
    }
}

/*==================[Main]======================*/
void app_main(void)
{
    motorsSetup();   //Inicializar el ESP32 para usar los motores
   
    // 1. Inicializar NVS (Memoria flash)
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    // 2. Inicializar TCP/IP
    ESP_ERROR_CHECK(esp_netif_init());

    // 3. Inicializar el loop de eventos
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    // 4. Crear interfaz de red WiFi AP
    wifi_init_softap();

    //xTaskCreate(tcp_server_task, "tcp_server", 4096, NULL, 5, NULL); // Crea la tarea del servidor TCP
    BaseType_t errC = xTaskCreatePinnedToCore(
        tcp_server_task,                    // Funcion de la tarea a ejecutar
        "tcp_server",   	                // Nombre de la tarea como String amigable para el usuario
        configMINIMAL_STACK_SIZE*4,         // Cantidad de stack de la tarea
        NULL,                          	    // Parametros de tarea
        tskIDLE_PRIORITY+5,         	    // Prioridad de la tarea -> Queremos que este un nivel encima de IDLE
        &wifi_task_handle,                          		// Puntero a la tarea creada en el sistema
        PROCESADORA                         // Numero de procesador
    );

    // Gestion de errores
    if(errC == pdFAIL)
    {
        ESP_LOGI(TAG, "Error al crear la tarea de la RED.");
        while(1);    // Si no pudo crear la tarea queda en un bucle infinito
    }   

    motor_state_mutex = xSemaphoreCreateMutex();
    
    if (motor_state_mutex == NULL) {
        ESP_LOGE(TAG, "Error al crear el mutex del motor");
        while(1);
    }

    BaseType_t errD = xTaskCreatePinnedToCore(
            motor_up_task,                      // Función de la tarea
            "motor_up",                         // Nombre de la tarea
            configMINIMAL_STACK_SIZE * 4,        // Tamaño del stack
            NULL,                      // Parámetro
            tskIDLE_PRIORITY + 4,                // Prioridad
            &motor_task_handle,                    // Handle (opcional)
            PROCESADORB                         // Núcleo
    );

     // Gestion de errores
    if(errD == pdFAIL)
    {
        ESP_LOGI(TAG, "Error al crear la tarea del MOTOR.");
        while(1);    // Si no pudo crear la tarea queda en un bucle infinito
    }  
}
