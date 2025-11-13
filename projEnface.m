function E = projEnface(ROI, depth_range, projection_type)
    % ROI: 3D matrix representing the OCT volume
    % depth_range: the range of depth (Z) for projection, e.g., 280:290
    % projection_type: type of projection, options: 'mean', 'max', 'var'
    
    % Extract the depth slice(s) from the ROI
    volume_slice = ROI(depth_range, :, :);  % Select the given depth range
    
    % Perform the specified projection along the depth direction (Z)
    switch projection_type
        case 'mean'
            % Compute the mean projection along the Z direction
            E = squeeze(mean(volume_slice, 1));  % Mean along the depth dimension
        case 'max'
            % Compute the maximum intensity projection along the Z direction
            E = squeeze(max(volume_slice, [], 1));  % Max along the depth dimension
        case 'var'
            % Compute the variance projection along the Z direction
            E = squeeze(var(volume_slice, 0, 1));  % Variance along the depth dimension
        otherwise
            error('Unsupported projection type. Use ''mean'', ''max'', or ''var''.');
    end
end