import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        System.out.println("=== Phishing Email Detector ===");
        System.out.print("Enter an email message: ");
        String email = scanner.nextLine();

        try {
            ProcessBuilder builder = new ProcessBuilder(
                ".venv/bin/python", "python/predict.py"
            );

            builder.redirectError(ProcessBuilder.Redirect.INHERIT);
            Process process = builder.start();

            try (OutputStreamWriter writer = new OutputStreamWriter(
                    process.getOutputStream(), StandardCharsets.UTF_8)) {
                writer.write(email);
            }

            String prediction;
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(
                        process.getInputStream(), StandardCharsets.UTF_8))) {
                prediction = reader.readLine();
            }

            int exitCode = process.waitFor();

            if (exitCode == 0 && prediction != null) {
                System.out.println("\nPrediction: " + prediction);
            } else {
                System.out.println("Python prediction failed.");
            }

        } catch (IOException e) {
            System.out.println("Could not start Python: " + e.getMessage());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            System.out.println("Prediction interrupted.");
        }

        scanner.close();
    }
}