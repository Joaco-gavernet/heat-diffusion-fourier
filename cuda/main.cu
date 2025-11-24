#include <stdio.h>
#include "heat_simulation.h"
#ifdef GLUT
#include <GL/freeglut.h>
#endif

#define WIDTH 640
#define HEIGHT 480

float zoom = 1.0f;
float offsetX = 0.0f;
float offsetY = 0.0f;

unsigned int ms = 30; // milisegundos - equivale a Frames por Segundo -> FPSs= 1000/ (milisegundos por frame) -> 30ms son 33 FPS, 100ms son 10FPS

/* 
   Para medir el tiempo transcurrido (elapsed time):

   resnfo: tipo de dato definido para abstraer la métrica de recursos a usar
   timenfo: tipo de dato definido para abstraer la métrica de tiempo a usar

   timestamp: abstrae función usada para tomar las muestras del tiempo transcurrido

   printtime: abstrae función usada para imprimir el tiempo transcurrido

   void myElapsedtime(resnfo start, resnfo end, timenfo *t): función para obtener 
   el tiempo transcurrido entre dos medidas
*/

#include <sys/time.h>
// #include <sys/resource.h>

double dwalltime(){
    double sec;
    struct timeval tv;

    gettimeofday(&tv,NULL);
    sec = tv.tv_sec + tv.tv_usec/1000000.0;
    return sec;
}

#ifdef GLUT
float clamp(float val, float min, float max) {
    if (val < min) return min;
    if (val > max) return max;
    return val;
}

void temperatura_a_color(float temp, float* r, float* g, float* b) {
    temp = clamp(temp, 0.0f, 100.0f);

    if (temp < 50.0f) {
        *r = 0.0f;
        *g = temp / 50.0f;
        *b = 1.0f - (*g);
    } else {
        *r = (temp - 50.0f) / 50.0f;
        *g = 1.0f - (*r);
        *b = 0.0f;
    }
}

void dibujarSuperficie(){
    float r, g, b;
    float cell_size = 1.0f; // Cada celda mide 1 unidad si se usa gluOrtho2D(0, grid_size, 0, grid_size)

    for (int y = 0; y < grid_size; y++) {
        for (int x = 0; x < grid_size; x++) {
            // Color de la celda (RGB normalizado a [0,1])
            temperatura_a_color(grid[y*grid_size + x], &r, &g, &b);
            glColor3f(r, g, b);

            // Dibujar el cuadrado de la celda
            glBegin(GL_QUADS);
            glVertex2f(y, x);
            glVertex2f(y + cell_size, x);
            glVertex2f(y + cell_size, x + cell_size);
            glVertex2f(y, x + cell_size);
            glEnd();
        }
    }
}

void display(void){
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    dibujarSuperficie();

    glutSwapBuffers();
}

void timer(int value) {
    update_simulation();
    glutPostRedisplay();
    glutTimerFunc(ms, timer, 0);
}

void updateProjection(int w, int h){
    glViewport(0, 0, w, h);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    float view_size = grid_size / zoom;

    gluOrtho2D(offsetX, offsetX + view_size, offsetY, offsetY + view_size);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void reshape(int w, int h){
    updateProjection(w,h);
}

void keyboardKeys(unsigned char key, int x, int y){
    if (key == 27) { //ESC
        destroy_grid();
        exit(0);
    }
}

void specialKeys(int key, int x, int y){
float move_step = grid_size * 0.05f / zoom;

    switch (key) {
        case GLUT_KEY_UP:    offsetY -= move_step; break;
        case GLUT_KEY_DOWN:  offsetY += move_step; break;
        case GLUT_KEY_LEFT:  offsetX += move_step; break;
        case GLUT_KEY_RIGHT: offsetX -= move_step; break;
    }

    updateProjection(WIDTH, HEIGHT);
    glutPostRedisplay();
}

void mouseWheel(int wheel, int direction, int x, int y){
    // Tama�o visible antes del zoom
    float view_size_old = grid_size / zoom;

    if (direction > 0)
        zoom *= 1.1f;   // acercar
    else
        zoom /= 1.1f;   // alejar

    if (zoom < 0.1f) zoom = 0.1f;

    // Tama�o visible despues del zoom
    float view_size_new = grid_size / zoom;

    // Centro actual antes del zoom
    float centerX = offsetX + view_size_old / 2.0f;
    float centerY = offsetY + view_size_old / 2.0f;

    // Modificar offset para mantener el centro en el mismo lugar
    offsetX = centerX - view_size_new / 2.0f;
    offsetY = centerY - view_size_new / 2.0f;

    updateProjection(WIDTH, HEIGHT);
    glutPostRedisplay();
}
#endif

int main(int argc, char* argv[]) {
    
    int N = 32;
    int cuda_block_size = 16;
    int max_steps = 1000;
    
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) cuda_block_size = atoi(argv[2]);
    if (argc > 3) max_steps = atoi(argv[3]);

    printf("(host) Parametros de simulacion: \n");
    printf("(host) N = %d\n", N);
    printf("(host) cuda_block_size = %d\n", cuda_block_size);
    printf("(host) max_steps = %d\n", max_steps);
    printf("(host) //////////////////////////////////\n");
    
    double timeini = dwalltime();
    initialize_grid(N, cuda_block_size);
    printf("(host) Tiempo de inicializacion de la grilla %f s\n", dwalltime() - timeini);
    
    double timesim = dwalltime(); 
    for (int i = 0; i < max_steps; i++) {
        if (update_simulation()) break;
    }
    #ifdef GLUT
    (ms, timer, 0); // Llama a timer (el paso de simulacion) cada ms milisegundos - FPSs= 1000/ (milisegundos por frame)
    #endif
    destroy_grid();    
    float total_time = dwalltime() - timeini; 
    float sim_time = dwalltime() - timesim;
    printf("(host) Tiempo de simulacion %f s\n", sim_time);
    printf("(host) Tiempo total (inicializacion y simulacion) %f s\n", total_time);
    printf("(host) Tiempo promedio de paso de simulacion %f s\n", sim_time /max_steps);

    return 0;
}
