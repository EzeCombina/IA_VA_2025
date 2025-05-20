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

/*==================[Definiciones]======================*/
#define PORT 3333
#define PROCESADORA 0
#define PROCESADORB 1

/*==================[Variables globales]======================*/
static const char *TAG = "TCP_SERVER";

int *p;

int client_sock = -1;  // global

QueueHandle_t comando_queue;

//SemaphoreHandle_t sock_mutex;  // Mutex para proteger el acceso al socket
//sock_mutex = xSemaphoreCreateMutex();

/*==================[Funciones]======================*/
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
        uint addr_len = sizeof(source_addr); // Longitud de la dirección
        // VER DE AGREGAR UN MUTEX PARA PROTEGER EL SOCKET
        client_sock = accept(listen_sock, (struct sockaddr *)&source_addr, &addr_len); // Acepta la conexión

        inet_ntoa_r(((struct sockaddr_in *)&source_addr)->sin_addr, addr_str, sizeof(addr_str) - 1); // Convierte la dirección IP a una cadena
        ESP_LOGI(TAG, "Cliente conectado desde %s", addr_str); // Imprime la dirección IP del cliente

        while(1)
        {
            int len = recv(client_sock, rx_buffer, sizeof(rx_buffer) - 1, 0);
            if(len < 0)
            {
                ESP_LOGE(TAG, "Error en recv");
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
                ESP_LOGI(TAG, "Comando recibido: %s", rx_buffer);
                if (xQueueSend(comando_queue, rx_buffer, pdMS_TO_TICKS(100)) != pdPASS) 
                {
                    ESP_LOGW(TAG, "No se pudo enviar el comando a la cola");
                }
            }
        }
        close(client_sock);
    }
}

void motores_task(void *pvParameters)
{
    while(1)
    {
        // Aquí se implementa la lógica para controlar los motores
        char comando[128];

        if (xQueueReceive(comando_queue, comando, portMAX_DELAY) == pdPASS) {
            ESP_LOGI(TAG, "Comando recibido por motores_task: %s", comando);

            if(strcmp(comando, "Adelante") == 0) {
                ESP_LOGI(TAG, "Movimiento: Adelante");
                send(client_sock, "Recibido", strlen("Recibido"), 0);
            }
            else if(strcmp(comando, "Atras") == 0) {
                ESP_LOGI(TAG, "Movimiento: Atras");
                send(client_sock, "Recibido", strlen("Recibido"), 0);
            }
            else if(strcmp(comando, "Derecha") == 0) {
                ESP_LOGI(TAG, "Movimiento: Derecha");
                send(client_sock, "Recibido", strlen("Recibido"), 0);
            }
            else if(strcmp(comando, "Izquierda") == 0) {
                ESP_LOGI(TAG, "Movimiento: Izquierda");
                send(client_sock, "Recibido", strlen("Recibido"), 0);
            }
        }
        vTaskDelay(1000 / portTICK_PERIOD_MS); // Simula un retardo de 1 segundo
    }
}

/*==================[Main]======================*/
void app_main(void)
{
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

    // Crear la cola de comandos
    comando_queue = xQueueCreate(5, sizeof(char) * 128); // Cola para 5 comandos de hasta 128 bytes
    if (comando_queue == NULL) {
        ESP_LOGE(TAG, "No se pudo crear la cola de comandos");
        while(1);
    }

    //xTaskCreate(tcp_server_task, "tcp_server", 4096, NULL, 5, NULL); // Crea la tarea del servidor TCP
    BaseType_t errA = xTaskCreatePinnedToCore(
        tcp_server_task,                    // Funcion de la tarea a ejecutar
        "tcp_server",   	                // Nombre de la tarea como String amigable para el usuario
        configMINIMAL_STACK_SIZE*4,         // Cantidad de stack de la tarea
        NULL,                          	    // Parametros de tarea
        tskIDLE_PRIORITY+5,         	    // Prioridad de la tarea -> Queremos que este un nivel encima de IDLE
        NULL,                          		// Puntero a la tarea creada en el sistema
        PROCESADORA                         // Numero de procesador
    );

    // Gestion de errores
    if(errA == pdFAIL)
    {
        ESP_LOGI(TAG, "Error al crear la tarea.");
        while(1);    // Si no pudo crear la tarea queda en un bucle infinito
    }

    BaseType_t errB = xTaskCreatePinnedToCore(
        motores_task,                    // Funcion de la tarea a ejecutar
        "motores_task",   	                // Nombre de la tarea como String amigable para el usuario
        configMINIMAL_STACK_SIZE*4,         // Cantidad de stack de la tarea
        NULL,                          	    // Parametros de tarea
        tskIDLE_PRIORITY+5,         	    // Prioridad de la tarea -> Queremos que este un nivel encima de IDLE
        NULL,                          		// Puntero a la tarea creada en el sistema
        PROCESADORA                         // Numero de procesador
    );

    if(errB == pdFAIL)
    {
        ESP_LOGI(TAG, "Error al crear la tarea.");
        while(1);    // Si no pudo crear la tarea queda en un bucle infinito
    }

    // Agregar una tarea que controle a los motores en función de los comandos recibidos
    // Agregar que cuando se termine de realizar el movimiento mande un mensaje al cliente
    // Agregar un watch dog en donde si no se recibe un comando en ... segundos se apagan los motores
}
