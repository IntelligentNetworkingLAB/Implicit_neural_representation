def image_boundary(model_output, gt, epoch) : 
    if epoch <= 100: 
        original = ((gt['img'] * 0.5 + 0.5) * 255).reshape(256, 256, 3).cpu().numpy().astype('uint8')
        #original = ((gt['img'].view(256, 256) * 0.5 + 0.5) * 255).cpu().numpy().astype('uint8')
        
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 150, 250)
        edges_filename = f'boundary.png'
        cv2.imwrite(edges_filename, edges)
        
        image_counter = torch.from_numpy(edges / 255.0).unsqueeze(-1).repeat(1, 1, 3).cuda().view(1, 65536, 3)
        #image_counter = image_counter * (-1.0) + 1.0     
        loss_counter = (image_counter * (model_output['model_out'] - gt['img']) ** 2).mean()

        return {'img_loss': loss_counter}
    
    else:
        loss = ((model_output['model_out'] - gt['img']) ** 2).mean()
        return {'img_loss': loss}
    