#include <Eigen/Dense>
#include <random>
#include <chrono>
#include <vector>
#include <iostream>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <implot.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

using namespace Eigen;


double generate_noise(double stddev)
{
    static std::default_random_engine gen(std::random_device{}());
    std::normal_distribution<double> dist(0.0, stddev);
    return dist(gen);
}

struct KalmanFilter1D
{
    double μₜ; // μₜ: State estimate
    double Σₜ; // Σₜ: Estimate covariance
    double Aₜ; // Aₜ: State transition
    double Rₜ; // Rₜ: Process noise covariance
    double Cₜ; // Cₜ: Observation model
    double Qₜ; // Qₜ: Measurement noise covariance
    double Kₜ; // Kₜ: Kalman gain

    KalmanFilter1D() : μₜ(20.0), Σₜ(0.0), Aₜ(1.0), Rₜ(4.0), Cₜ(1.0), Qₜ(4.0), Kₜ(0.0) {}

    double predict(double update)
    {
        double ϵₜ = generate_noise(std::sqrt(Rₜ)); // processruis, ruis die in actuator zit.
        μₜ = μₜ + update + ϵₜ; // Voorspel de nieuwe staat
        Σₜ = Σₜ + Rₜ;     // Verhoog de onzekerheid door procesruis
        return μₜ;
    }

    void gain()
    {
        double upper = Σₜ;
        double lower = Σₜ + Qₜ;
        Kₜ = upper / lower;
    }

    double adjust(double Zₜ)
    {
        μₜ = μₜ + Kₜ * (Zₜ - μₜ);
        Σₜ = (1.0 - Kₜ) * Σₜ;
        return μₜ;
    }
};

int main()
{
    if (!glfwInit())
    {
        std::cerr << "Fout bij initialisatie van GLFW\n";
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Kalman Filter Visualisatie", nullptr, nullptr);
    if (!window)
    {
        std::cerr << "Fout bij het aanmaken van het venster\n";
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // V-sync aan

    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    KalmanFilter1D kf;

    std::vector<double> time_points, true_vals, measured_vals, predicted_vals, adjusted_vals, kalman_gains;
    const size_t max_points = 200;

    auto push_and_trim = [](std::vector<double>& vec, double val, size_t max_points) {
        vec.push_back(val);
        if (vec.size() > max_points)
            vec.erase(vec.begin());
    };

    double true_temp = 20.0;
    double t = 0.0;
    const double dt = 1.0;
    const double update_interval = 0.1;

    double measurementActualNoise = 5.0;
    double updateActualNoise = 0.5;
    auto last_time = std::chrono::high_resolution_clock::now();

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - last_time).count();

        if (elapsed >= update_interval)
        {
            last_time = now;
            t += dt;

            true_temp += 0.2;

            double measurementWithNoise = generate_noise(std::sqrt(measurementActualNoise));
            double updateWithNoise = generate_noise(std::sqrt(updateActualNoise));

            double Zₜ = true_temp + measurementWithNoise;

            double prediction = kf.predict(0.2 + updateWithNoise);

            kf.gain();
            double adjustedStateVector = kf.adjust(Zₜ);

            push_and_trim(time_points, t, max_points);
            push_and_trim(true_vals, true_temp, max_points);
            push_and_trim(measured_vals, Zₜ, max_points);
            push_and_trim(predicted_vals, prediction, max_points);
            push_and_trim(adjusted_vals, adjustedStateVector, max_points);
            push_and_trim(kalman_gains, kf.Kₜ, max_points);
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Kalman Filter Instellingen");
        // prediction sliders
        double min = 0.0, max = 10.0;
        ImGui::SliderScalar("Daadwerkelijke proces afwijking", ImGuiDataType_Double, &updateActualNoise, &min, &max);
        ImGui::SliderScalar("Gegeven afwijking Rₜ", ImGuiDataType_Double, &kf.Rₜ, &min, &max);

        // measurement sliders
        ImGui::SliderScalar("Daadwerkelijke sensor afwijking", ImGuiDataType_Double, &measurementActualNoise, &min, &max);
        ImGui::SliderScalar("Gegeven afwijking Qₜ", ImGuiDataType_Double, &kf.Qₜ, &min, &max);
        ImGui::End();

        if (ImPlot::BeginPlot("Temperatuur Tracking", ImVec2(-1, 300)))
        {
            ImPlot::SetupAxes("Tijd (s)", "Temperatuur (°C)", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::PlotLine("Echte temperatuur", time_points.data(), true_vals.data(), time_points.size());
            ImPlot::PlotLine("predected state", time_points.data(), predicted_vals.data(), time_points.size());
            ImPlot::PlotLine("measurement", time_points.data(), measured_vals.data(), time_points.size());
            ImPlot::PlotLine("adjusted state", time_points.data(), adjusted_vals.data(), time_points.size());
            ImPlot::EndPlot();
        }

        if (ImPlot::BeginPlot("Kalman Gain", ImVec2(-1, 200)))
        {
            ImPlot::SetupAxes("Tijd (s)", "Gain", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::PlotLine("Kalman Gain", time_points.data(), kalman_gains.data(), time_points.size());
            ImPlot::EndPlot();
        }

        ImGui::Render();
        glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
